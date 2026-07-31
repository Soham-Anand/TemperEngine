#include "temper/memory/scheduler.h"
#include "temper/memory/compression.h"
#include "temper/core/device.h"
#include "temper/core/runtime.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TierState
{
    size_t budget;    // max bytes allowed
    size_t used;      // resident footprint (compressed_size if compressed, else bytes)
    size_t logical;   // uncompressed bytes of every registered resource in this tier
    size_t reserved;  // footprint of pinned resources (cannot evict)
} TierState;

struct TemperMemScheduler
{
    uint32_t version;
    bool initialized;
    TierState tiers[TEMPER_TIER_COUNT];
    TemperResource **resources;
    uint32_t resource_count;
    uint32_t resource_capacity;
    float recompute_threshold;
    float pressure_low;
    float pressure_high;
    uint64_t stats[TEMPER_STAT_COUNT];
    const TemperCompressionBackend *backend;
};

static TemperMemScheduler g_sched;

static bool tier_valid(TemperMemTier tier)
{
    return (unsigned)tier < (unsigned)TEMPER_TIER_COUNT;
}

// Current resident footprint of a resource (0 if not resident).
static size_t resource_footprint(const TemperResource *res)
{
    if (!(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        return 0;
    }
    if (res->flags & TEMPER_RESOURCE_COMPRESSED)
    {
        return res->compressed_size;
    }
    return res->bytes;
}

// Invariant: flags and backing stores must be internally consistent.
static bool resource_consistent(const TemperResource *res)
{
    if (!res)
    {
        return false;
    }
    if ((res->flags & TEMPER_RESOURCE_COMPRESSED) && !(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        return false;
    }
    if (!(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        if (res->host_ptr != NULL || res->compressed_blob != NULL || res->compressed_size != 0)
        {
            return false;
        }
    }
    else if (res->flags & TEMPER_RESOURCE_COMPRESSED)
    {
        if (res->compressed_blob == NULL || res->compressed_size == 0 || res->host_ptr != NULL)
        {
            return false;
        }
    }
    else
    {
        if (res->host_ptr == NULL || res->compressed_blob != NULL || res->compressed_size != 0)
        {
            return false;
        }
    }
    return true;
}

// Move accounting for a resource changing tier with a footprint change.
// Must be called with the resource's footprint fields already updated for `to`.
static void move_accounting(TemperResource *res, TemperMemTier from, TemperMemTier to,
                            size_t from_footprint, size_t to_footprint)
{
    TierState *fs = &g_sched.tiers[from];
    TierState *ts = &g_sched.tiers[to];
    fs->used -= from_footprint;
    fs->logical -= res->bytes;
    ts->used += to_footprint;
    ts->logical += res->bytes;
    if (res->flags & TEMPER_RESOURCE_PINNED)
    {
        fs->reserved -= from_footprint;
        ts->reserved += to_footprint;
    }
}

// Free a resource's storage, keeping its logical accounting (recompute drop).
static void drop_resource(TemperResource *res)
{
    TierState *ts = &g_sched.tiers[res->tier];
    size_t foot = resource_footprint(res);
    if (res->flags & TEMPER_RESOURCE_PINNED)
    {
        ts->reserved -= foot;
    }
    ts->used -= foot;
    if (res->compressed_blob)
    {
        free(res->compressed_blob);
        res->compressed_blob = NULL;
    }
    if (res->host_ptr)
    {
        if (res->allocator)
        {
            res->allocator->free_host(res->host_ptr);
        }
        else
        {
            free(res->host_ptr);
        }
        res->host_ptr = NULL;
        res->native = NULL;
    }
    res->compressed_size = 0;
    res->flags &= ~(TEMPER_RESOURCE_RESIDENT | TEMPER_RESOURCE_COMPRESSED);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void temper_scheduler_init(void)
{
    if (g_sched.initialized)
    {
        return;
    }
    memset(&g_sched, 0, sizeof(g_sched));
    g_sched.version = TEMPER_SCHEDULER_VERSION;
    g_sched.initialized = true;

    size_t cpu_default = (size_t)1 << 30; // 1 GiB logical default

    // GPU tier: prefer registered GPU memory, else mirror the CPU default.
    size_t gpu_budget = 0;
    const TemperDeviceTable *table = temper_device_table_get();
    for (uint32_t i = 0; i < table->count; i++)
    {
        if (temper_device_is_gpu(table->devices[i]))
        {
            gpu_budget += table->caps[i].total_memory;
        }
    }
    if (gpu_budget == 0)
    {
        gpu_budget = cpu_default;
    }

    g_sched.tiers[TEMPER_TIER_GPU].budget = gpu_budget;
    g_sched.tiers[TEMPER_TIER_CPU].budget = cpu_default;
    g_sched.tiers[TEMPER_TIER_COMPRESSED].budget = cpu_default * 2;
    g_sched.tiers[TEMPER_TIER_SSD].budget = (size_t)1 << 38; // 256 GiB logical ceiling

    g_sched.recompute_threshold = 0.5f;
    g_sched.pressure_low = 0.5f;
    g_sched.pressure_high = 0.85f;
    g_sched.backend = &temper_compression_bf16;
}

void temper_scheduler_shutdown(void)
{
    if (!g_sched.initialized)
    {
        return;
    }
    if (g_sched.resource_count > 0)
    {
        temper_warn("Scheduler shutdown with %u registered resources", g_sched.resource_count);
    }
    free(g_sched.resources);
    g_sched.resources = NULL;
    g_sched.resource_count = 0;
    g_sched.resource_capacity = 0;
    g_sched.initialized = false;
}

bool temper_scheduler_initialized(void)
{
    return g_sched.initialized;
}

const TemperMemScheduler *temper_scheduler_get(void)
{
    return &g_sched;
}

uint32_t temper_scheduler_version(void)
{
    return g_sched.version;
}

TemperMemTier temper_scheduler_tier_for_device(TemperDevice device)
{
    return temper_device_is_gpu(device) ? TEMPER_TIER_GPU : TEMPER_TIER_CPU;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void temper_scheduler_set_tier_budget(TemperMemTier tier, size_t bytes)
{
    if (tier_valid(tier))
    {
        g_sched.tiers[tier].budget = bytes;
    }
}

size_t temper_scheduler_tier_budget(TemperMemTier tier)
{
    return tier_valid(tier) ? g_sched.tiers[tier].budget : 0;
}

size_t temper_scheduler_tier_used(TemperMemTier tier)
{
    return tier_valid(tier) ? g_sched.tiers[tier].used : 0;
}

size_t temper_scheduler_tier_logical(TemperMemTier tier)
{
    return tier_valid(tier) ? g_sched.tiers[tier].logical : 0;
}

size_t temper_scheduler_tier_reserved(TemperMemTier tier)
{
    return tier_valid(tier) ? g_sched.tiers[tier].reserved : 0;
}

void temper_scheduler_set_thresholds(float recompute_threshold, float pressure_low, float pressure_high)
{
    g_sched.recompute_threshold = recompute_threshold;
    g_sched.pressure_low = pressure_low;
    g_sched.pressure_high = pressure_high;
}

// ---------------------------------------------------------------------------
// Pressure
// ---------------------------------------------------------------------------

float temper_scheduler_pressure(TemperMemTier tier)
{
    if (!tier_valid(tier))
    {
        return 1.0f;
    }
    if (g_sched.tiers[tier].budget == 0)
    {
        return 1.0f;
    }
    return (float)g_sched.tiers[tier].used / (float)g_sched.tiers[tier].budget;
}

bool temper_scheduler_under_pressure(TemperMemTier tier)
{
    return temper_scheduler_pressure(tier) > g_sched.pressure_high;
}

// ---------------------------------------------------------------------------
// Policy: scoring (standalone, replaceable)
// ---------------------------------------------------------------------------

float temper_recompute_score(const TemperResource *res)
{
    if (!res || !(res->flags & TEMPER_RESOURCE_RECOMPUTABLE))
    {
        return -1.0f;
    }
    // Phase 3 proxy: recompute cost == bytes (ADR-006 Amendment 3).
    // Phase 7 wires real per-op FLOP costs via the origin node.
    float memory_saved = (float)res->bytes;
    float recompute_cost = (float)res->bytes;
    if (recompute_cost < 1.0f)
    {
        recompute_cost = 1.0f;
    }
    // Deterministic access factor (ADR-006 Amendment 2): never-accessed wins.
    float access_factor = 1.0f / (1.0f + (float)res->access_count);
    return (memory_saved / recompute_cost) * access_factor;
}

float temper_placement_score(const TemperResource *res, TemperDevice device)
{
    if (!res)
    {
        return 0.0f;
    }
    float score = 0.0f;
    float access_rate = (float)res->access_count / (float)(res->access_count + 1);
    score += access_rate * 0.3f;
    if (res->flags & TEMPER_RESOURCE_RECOMPUTABLE)
    {
        score -= 0.2f; // regenerable tensors are lower placement priority
    }
    if (!temper_device_equal(res->device, device))
    {
        score -= 0.2f; // device transfer penalty
    }
    score -= temper_scheduler_pressure(temper_scheduler_tier_for_device(device)) * 0.3f;
    return score;
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

int temper_scheduler_reserve(TemperDevice device, size_t bytes)
{
    if (!g_sched.initialized)
    {
        return 0;
    }
    TemperMemTier tier = temper_scheduler_tier_for_device(device);
    if (g_sched.tiers[tier].used + bytes <= g_sched.tiers[tier].budget)
    {
        return 0;
    }
    temper_scheduler_evict(tier, bytes);
    size_t now = g_sched.tiers[tier].used + bytes;
    if (now > g_sched.tiers[tier].budget)
    {
        temper_warn("Tier %d over budget after eviction (%zu/%zu bytes)",
                    (int)tier, now, g_sched.tiers[tier].budget);
        return (int)(now - g_sched.tiers[tier].budget);
    }
    return 0;
}

void temper_scheduler_register(TemperResource *res)
{
    if (!g_sched.initialized || !res)
    {
        return;
    }
    if (g_sched.resource_count == g_sched.resource_capacity)
    {
        uint32_t new_cap = g_sched.resource_capacity ? g_sched.resource_capacity * 2 : 16;
        TemperResource **new_arr = (TemperResource **)realloc(g_sched.resources,
                                                              sizeof(TemperResource *) * new_cap);
        TEMPER_ASSERT_MSG(new_arr != NULL, "Scheduler resource table grow failed");
        g_sched.resources = new_arr;
        g_sched.resource_capacity = new_cap;
    }
    g_sched.resources[g_sched.resource_count++] = res;

    TierState *ts = &g_sched.tiers[res->tier];
    size_t foot = resource_footprint(res);
    ts->used += foot;
    ts->logical += res->bytes;
    if (res->flags & TEMPER_RESOURCE_PINNED)
    {
        ts->reserved += foot;
    }
}

void temper_scheduler_unregister(TemperResource *res)
{
    if (!g_sched.initialized || !res)
    {
        return;
    }
    for (uint32_t i = 0; i < g_sched.resource_count; i++)
    {
        if (g_sched.resources[i] == res)
        {
            TierState *ts = &g_sched.tiers[res->tier];
            size_t foot = resource_footprint(res);
            ts->used -= foot;
            ts->logical -= res->bytes;
            if (res->flags & TEMPER_RESOURCE_PINNED)
            {
                ts->reserved -= foot;
            }
            g_sched.resources[i] = g_sched.resources[g_sched.resource_count - 1];
            g_sched.resource_count--;
            break;
        }
    }
}

void temper_scheduler_on_tier_change(TemperResource *res, TemperMemTier from, TemperMemTier to)
{
    if (!g_sched.initialized || !res || from == to)
    {
        return;
    }
    size_t foot = resource_footprint(res);
    move_accounting(res, from, to, foot, foot);
}

// ---------------------------------------------------------------------------
// Eviction (mechanism layer)
// ---------------------------------------------------------------------------

int temper_scheduler_evict(TemperMemTier tier, size_t bytes_needed)
{
    if (!g_sched.initialized || bytes_needed == 0 || !tier_valid(tier))
    {
        return 0;
    }
    TierState *ts = &g_sched.tiers[tier];

    TemperResource **candidates = (TemperResource **)malloc(
        g_sched.resource_count ? sizeof(TemperResource *) * g_sched.resource_count : 1);
    if (!candidates)
    {
        return -1;
    }
    uint32_t count = 0;
    for (uint32_t i = 0; i < g_sched.resource_count; i++)
    {
        TemperResource *r = g_sched.resources[i];
        if (r->tier != tier)
        {
            continue;
        }
        if (r->flags & TEMPER_RESOURCE_PINNED)
        {
            continue;
        }
        candidates[count++] = r;
    }

    // Deterministic LRU: oldest last_access first, ties broken on resource id.
    for (uint32_t i = 1; i < count; i++)
    {
        TemperResource *key = candidates[i];
        uint32_t j = i;
        while (j > 0)
        {
            TemperResource *prev = candidates[j - 1];
            bool earlier = key->last_access < prev->last_access ||
                           (key->last_access == prev->last_access && key->id < prev->id);
            if (!earlier)
            {
                break;
            }
            candidates[j] = prev;
            j--;
        }
        candidates[j] = key;
    }

    for (uint32_t i = 0; i < count; i++)
    {
        if (ts->used + bytes_needed <= ts->budget)
        {
            break;
        }
        TemperResource *victim = candidates[i];

        // Branch 1: recompute beats storage.
        if (temper_recompute_score(victim) > g_sched.recompute_threshold)
        {
            drop_resource(victim);
            g_sched.stats[TEMPER_STAT_RECOMPUTATIONS]++;
            g_sched.stats[TEMPER_STAT_EVICTIONS]++;
            continue;
        }
        // Branch 2: demote to a lower tier.
        if (temper_resource_demote(victim) == 0)
        {
            g_sched.stats[TEMPER_STAT_EVICTIONS]++;
            continue;
        }
        // Cannot evict this victim; try the next candidate.
    }

    free(candidates);
    if (ts->used + bytes_needed <= ts->budget)
    {
        return 0;
    }
    temper_warn("Eviction from tier %d could not free the requested %zu bytes", (int)tier, bytes_needed);
    return (int)(ts->used + bytes_needed - ts->budget);
}

// ---------------------------------------------------------------------------
// Movement: demotion / promotion
// ---------------------------------------------------------------------------

int temper_resource_demote(TemperResource *res)
{
    if (!res || !(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        return -1;
    }
    TemperMemTier from = res->tier;

    if (from == TEMPER_TIER_GPU)
    {
        // GPU -> CPU: device-tag change only (CPU-first engine, same host buffer).
        size_t foot = resource_footprint(res);
        move_accounting(res, from, TEMPER_TIER_CPU, foot, foot);
        res->tier = TEMPER_TIER_CPU;
        res->device = TEMPER_DEVICE_CPU_0;
        g_sched.stats[TEMPER_STAT_DEMOTIONS]++;
        return 0;
    }

    if (from == TEMPER_TIER_CPU)
    {
        // CPU -> COMPRESSED: physical compression via the backend.
        if (!g_sched.backend || !g_sched.backend->can_compress(res) || !g_sched.backend->compress)
        {
            return -1;
        }
        size_t element_count = res->bytes / sizeof(float);
        if (element_count == 0)
        {
            return -1;
        }
        size_t blob_size = 0;
        void *blob = g_sched.backend->compress(res->host_ptr, element_count, &blob_size);
        if (!blob)
        {
            return -1;
        }
        if (g_sched.tiers[TEMPER_TIER_COMPRESSED].used + blob_size > g_sched.tiers[TEMPER_TIER_COMPRESSED].budget)
        {
            free(blob);
            return -1; // compressed tier full; caller tries another victim
        }
        if (res->allocator)
        {
            res->allocator->free_host(res->host_ptr);
        }
        else
        {
            free(res->host_ptr);
        }
        res->host_ptr = NULL;
        res->native = NULL;
        res->compressed_blob = blob;
        res->compressed_size = blob_size;
        res->flags |= TEMPER_RESOURCE_COMPRESSED;
        move_accounting(res, from, TEMPER_TIER_COMPRESSED, res->bytes, blob_size);
        res->tier = TEMPER_TIER_COMPRESSED;
        res->device = TEMPER_DEVICE_CPU_0;
        g_sched.stats[TEMPER_STAT_COMPRESSIONS]++;
        g_sched.stats[TEMPER_STAT_DEMOTIONS]++;
        return 0;
    }

    if (from == TEMPER_TIER_COMPRESSED)
    {
        // COMPRESSED -> SSD: budget-accounted only in Phase 3 (no physical paging yet).
        size_t foot = resource_footprint(res);
        if (g_sched.tiers[TEMPER_TIER_SSD].used + foot > g_sched.tiers[TEMPER_TIER_SSD].budget)
        {
            return -1;
        }
        move_accounting(res, from, TEMPER_TIER_SSD, foot, foot);
        res->tier = TEMPER_TIER_SSD;
        g_sched.stats[TEMPER_STAT_DEMOTIONS]++;
        return 0;
    }

    return -1; // SSD is the floor; no physical paging until Phase 7
}

int temper_resource_promote(TemperResource *res)
{
    if (!res)
    {
        return -1;
    }
    if (res->flags & TEMPER_RESOURCE_COMPRESSED)
    {
        if (!g_sched.backend || !g_sched.backend->decompress)
        {
            return -1;
        }
        size_t element_count = res->bytes / sizeof(float);
        float *data = g_sched.backend->decompress(res->compressed_blob, res->compressed_size, element_count);
        if (!data)
        {
            return -1;
        }
        // Route storage back through the resource's allocator so ownership rules
        // hold regardless of backend (Metal shared buffer vs calloc).
        float *host = NULL;
        if (res->allocator)
        {
            host = (float *)res->allocator->alloc_host(res->bytes);
        }
        else
        {
            host = (float *)calloc(1, res->bytes);
        }
        if (!host)
        {
            free(data);
            return -1;
        }
        memcpy(host, data, res->bytes);
        free(data);
        size_t old_foot = res->compressed_size;
        move_accounting(res, res->tier, TEMPER_TIER_CPU, old_foot, res->bytes);
        res->tier = TEMPER_TIER_CPU;
        res->device = TEMPER_DEVICE_CPU_0;
        free(res->compressed_blob);
        res->compressed_blob = NULL;
        res->compressed_size = 0;
        res->host_ptr = host;
        res->native = host;
        res->flags |= TEMPER_RESOURCE_RESIDENT;
        res->flags &= ~TEMPER_RESOURCE_COMPRESSED;
        g_sched.stats[TEMPER_STAT_DECOMPRESSIONS]++;
        g_sched.stats[TEMPER_STAT_PROMOTIONS]++;
        return 0;
    }
    if (!(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        if (res->flags & TEMPER_RESOURCE_RECOMPUTABLE)
        {
            temper_warn("Recompute replay not implemented until Phase 7; resource %u unavailable", res->id);
        }
        else
        {
            temper_warn("Resource %u is non-resident and not recomputable; data unavailable", res->id);
        }
        return -1;
    }
    return 0; // already resident
}

// ---------------------------------------------------------------------------
// Pinning
// ---------------------------------------------------------------------------

void temper_scheduler_pin(TemperResource *res)
{
    if (!g_sched.initialized || !res)
    {
        return;
    }
    if (res->flags & TEMPER_RESOURCE_PINNED)
    {
        return;
    }
    res->flags |= TEMPER_RESOURCE_PINNED;
    if (res->flags & TEMPER_RESOURCE_RESIDENT)
    {
        g_sched.tiers[res->tier].reserved += resource_footprint(res);
    }
}

void temper_scheduler_unpin(TemperResource *res)
{
    if (!g_sched.initialized || !res)
    {
        return;
    }
    if (!(res->flags & TEMPER_RESOURCE_PINNED))
    {
        return;
    }
    res->flags &= ~TEMPER_RESOURCE_PINNED;
    if (res->flags & TEMPER_RESOURCE_RESIDENT)
    {
        g_sched.tiers[res->tier].reserved -= resource_footprint(res);
    }
}

// ---------------------------------------------------------------------------
// Compression backend
// ---------------------------------------------------------------------------

void temper_scheduler_set_compression_backend(const TemperCompressionBackend *backend)
{
    g_sched.backend = backend;
}

const TemperCompressionBackend *temper_scheduler_compression_backend(void)
{
    return g_sched.backend;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

uint64_t temper_scheduler_stat(TemperSchedulerStat stat)
{
    if ((unsigned)stat >= (unsigned)TEMPER_STAT_COUNT)
    {
        return 0;
    }
    return g_sched.stats[stat];
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

void temper_scheduler_dump_state(void)
{
    static const char *names[TEMPER_TIER_COUNT] = {"GPU", "CPU", "COMPRESSED", "SSD"};
    printf("TemperMemScheduler v%u\n", g_sched.version);
    for (int t = 0; t < TEMPER_TIER_COUNT; t++)
    {
        printf("  %-10s used=%8zu logical=%8zu budget=%8zu reserved=%8zu pressure=%.2f\n",
               names[t], g_sched.tiers[t].used, g_sched.tiers[t].logical,
               g_sched.tiers[t].budget, g_sched.tiers[t].reserved,
               (double)temper_scheduler_pressure((TemperMemTier)t));
    }
    printf("  stats: evictions=%llu recomputations=%llu compressions=%llu "
           "decompressions=%llu promotions=%llu demotions=%llu\n",
           (unsigned long long)g_sched.stats[TEMPER_STAT_EVICTIONS],
           (unsigned long long)g_sched.stats[TEMPER_STAT_RECOMPUTATIONS],
           (unsigned long long)g_sched.stats[TEMPER_STAT_COMPRESSIONS],
           (unsigned long long)g_sched.stats[TEMPER_STAT_DECOMPRESSIONS],
           (unsigned long long)g_sched.stats[TEMPER_STAT_PROMOTIONS],
           (unsigned long long)g_sched.stats[TEMPER_STAT_DEMOTIONS]);
    printf("  resources: %u registered\n", g_sched.resource_count);
}

bool temper_scheduler_validate(void)
{
    if (!g_sched.initialized)
    {
        return true;
    }
    if (g_sched.version != TEMPER_SCHEDULER_VERSION)
    {
        return false;
    }

    size_t used_sum[TEMPER_TIER_COUNT] = {0};
    size_t logical_sum[TEMPER_TIER_COUNT] = {0};
    size_t reserved_sum[TEMPER_TIER_COUNT] = {0};

    for (uint32_t i = 0; i < g_sched.resource_count; i++)
    {
        const TemperResource *res = g_sched.resources[i];
        if (!resource_consistent(res))
        {
            return false;
        }
        if (!tier_valid(res->tier))
        {
            return false;
        }
        size_t foot = resource_footprint(res);
        used_sum[res->tier] += foot;
        logical_sum[res->tier] += res->bytes;
        if (res->flags & TEMPER_RESOURCE_PINNED)
        {
            reserved_sum[res->tier] += foot;
        }
    }

    for (int t = 0; t < TEMPER_TIER_COUNT; t++)
    {
        if (g_sched.tiers[t].used != used_sum[t])
        {
            return false;
        }
        if (g_sched.tiers[t].logical != logical_sum[t])
        {
            return false;
        }
        if (g_sched.tiers[t].reserved != reserved_sum[t])
        {
            return false;
        }
        if (g_sched.tiers[t].used > g_sched.tiers[t].logical)
        {
            return false;
        }
        if (g_sched.tiers[t].reserved > g_sched.tiers[t].used)
        {
            return false;
        }
        if (g_sched.tiers[t].used > g_sched.tiers[t].budget)
        {
            return false;
        }
    }
    return true;
}
