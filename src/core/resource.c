#include "temper/core/resource.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/utils/assert.h"
#include <stdlib.h>
#include <string.h>

static uint32_t g_next_resource_id = 1;

TemperResource *temper_resource_create(TemperDevice device, size_t bytes)
{
    TemperResource *res = (TemperResource *)calloc(1, sizeof(TemperResource));
    TEMPER_ASSERT_MSG(res != NULL, "Failed to allocate TemperResource struct");

    res->id = g_next_resource_id++;
    res->device = device;
    res->bytes = bytes;
    res->refcount = 1;
    res->last_access = temper_time_us();
    res->pinned = false;
    res->recomputable = false;
    res->origin = NULL;

    if (bytes > 0)
    {
        res->host_ptr = (float *)calloc(1, bytes);
        TEMPER_ASSERT_MSG(res->host_ptr != NULL, "Failed to allocate host buffer for TemperResource");
        res->native = res->host_ptr;
    }
    else
    {
        res->host_ptr = NULL;
        res->native = NULL;
    }

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
        return 0; // Already on target device
    }

    // In Phase 2, migration creates/maintains host memory allocation and updates target device tag.
    // Future backends (Metal/CUDA) will transfer to MTLBuffer/cudaMalloc buffers here.
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
    }
}
