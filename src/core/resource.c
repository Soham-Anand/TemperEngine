#include "temper/core/resource.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/memory/scheduler.h"
#include "temper/utils/assert.h"
#include <stdlib.h>
#include <string.h>

static uint32_t g_next_resource_id = 1;

TemperResource *temper_resource_create(TemperDevice device, size_t bytes)
{
    // Give the scheduler a chance to evict before we allocate.
    temper_scheduler_reserve(device, bytes);

    TemperResource *res = (TemperResource *)calloc(1, sizeof(TemperResource));
    TEMPER_ASSERT_MSG(res != NULL, "Failed to allocate TemperResource struct");

    res->id = g_next_resource_id++;
    res->device = device;
    res->bytes = bytes;
    res->refcount = 1;
    res->flags = 0;
    res->tier = temper_scheduler_tier_for_device(device);
    res->access_count = 0;
    res->created_at = temper_time_us();
    res->last_access = res->created_at;
    res->origin = NULL;
    res->compressed_blob = NULL;
    res->compressed_size = 0;

    if (bytes > 0)
    {
        res->host_ptr = (float *)calloc(1, bytes);
        TEMPER_ASSERT_MSG(res->host_ptr != NULL, "Failed to allocate host buffer for TemperResource");
        res->native = res->host_ptr;
        res->flags |= TEMPER_RESOURCE_RESIDENT;
    }
    else
    {
        res->host_ptr = NULL;
        res->native = NULL;
    }

    temper_scheduler_register(res);
    return res;
}

void temper_resource_retain(TemperResource *res)
{
    if (!res)
    {
        return;
    }
    res->refcount++;
}

void temper_resource_release(TemperResource *res)
{
    if (!res)
    {
        return;
    }
    TEMPER_ASSERT(res->refcount > 0);
    res->refcount--;
    if (res->refcount == 0)
    {
        temper_scheduler_unregister(res);
        if (res->compressed_blob)
        {
            free(res->compressed_blob);
            res->compressed_blob = NULL;
        }
        if (res->host_ptr)
        {
            free(res->host_ptr);
            res->host_ptr = NULL;
            res->native = NULL;
        }
        free(res);
    }
}

int temper_resource_migrate(TemperResource *res, TemperDevice target_device)
{
    if (!res)
    {
        return -1;
    }
    if (temper_device_equal(res->device, target_device))
    {
        return 0;
    }
    if (!(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        temper_warn("Cannot migrate resource %u: not resident", res->id);
        return -1;
    }
    if (res->flags & TEMPER_RESOURCE_COMPRESSED)
    {
        if (temper_resource_promote(res) != 0)
        {
            return -1;
        }
    }

    TemperMemTier from = res->tier;
    TemperMemTier to = temper_scheduler_tier_for_device(target_device);
    if (from != to)
    {
        temper_scheduler_on_tier_change(res, from, to);
        res->tier = to;
    }
    res->device = target_device;
    res->last_access = temper_time_us();
    temper_info("Migrated resource %u to device %s", res->id, temper_device_name(target_device));
    return 0;
}

void temper_resource_touch(TemperResource *res)
{
    if (res)
    {
        res->last_access = temper_time_us();
        res->access_count++;
    }
}

bool temper_resource_is_resident(const TemperResource *res)
{
    return res != NULL && (res->flags & TEMPER_RESOURCE_RESIDENT) != 0;
}

bool temper_resource_is_compressed(const TemperResource *res)
{
    return res != NULL && (res->flags & TEMPER_RESOURCE_COMPRESSED) != 0;
}

bool temper_resource_is_recomputable(const TemperResource *res)
{
    return res != NULL && (res->flags & TEMPER_RESOURCE_RECOMPUTABLE) != 0;
}

bool temper_resource_is_pinned(const TemperResource *res)
{
    return res != NULL && (res->flags & TEMPER_RESOURCE_PINNED) != 0;
}
