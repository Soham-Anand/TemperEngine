#ifndef TEMPER_SCHEDULER_H
#define TEMPER_SCHEDULER_H

#include "temper/core/device.h"
#include "temper/core/resource.h"
#include "temper/memory/compression.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TEMPER_SCHEDULER_VERSION 1

typedef struct TemperMemScheduler TemperMemScheduler;

// Lifecycle
void temper_scheduler_init(void);
void temper_scheduler_shutdown(void);
bool temper_scheduler_initialized(void);
const TemperMemScheduler *temper_scheduler_get(void);
uint32_t temper_scheduler_version(void);

// Device <-> tier mapping (NPU treated as CPU tier until a backend exists)
TemperMemTier temper_scheduler_tier_for_device(TemperDevice device);

// Configuration
void temper_scheduler_set_tier_budget(TemperMemTier tier, size_t bytes);
size_t temper_scheduler_tier_budget(TemperMemTier tier);
size_t temper_scheduler_tier_used(TemperMemTier tier);
size_t temper_scheduler_tier_logical(TemperMemTier tier);
size_t temper_scheduler_tier_reserved(TemperMemTier tier);
void temper_scheduler_set_thresholds(float recompute_threshold, float pressure_low, float pressure_high);

// Pressure tracking (used = resident footprint, logical = uncompressed graph bytes)
float temper_scheduler_pressure(TemperMemTier tier);
bool temper_scheduler_under_pressure(TemperMemTier tier);

// Policy: standalone scoring functions (replaceable, separate from dispatch)
float temper_placement_score(const TemperResource *res, TemperDevice device);
float temper_recompute_score(const TemperResource *res);

// Resource management (called by resource.c)
int temper_scheduler_reserve(TemperDevice device, size_t bytes);
void temper_scheduler_register(TemperResource *res);
void temper_scheduler_unregister(TemperResource *res);
void temper_scheduler_on_tier_change(TemperResource *res, TemperMemTier from, TemperMemTier to);

// Movement
int temper_scheduler_evict(TemperMemTier tier, size_t bytes_needed);
int temper_resource_promote(TemperResource *res);
int temper_resource_demote(TemperResource *res);

// Pinning
void temper_scheduler_pin(TemperResource *res);
void temper_scheduler_unpin(TemperResource *res);

// Compression backend (policy never knows how compression works)
void temper_scheduler_set_compression_backend(const TemperCompressionBackend *backend);
const TemperCompressionBackend *temper_scheduler_compression_backend(void);

// Statistics
typedef enum TemperSchedulerStat
{
    TEMPER_STAT_EVICTIONS,
    TEMPER_STAT_RECOMPUTATIONS,
    TEMPER_STAT_COMPRESSIONS,
    TEMPER_STAT_DECOMPRESSIONS,
    TEMPER_STAT_PROMOTIONS,
    TEMPER_STAT_DEMOTIONS,
    TEMPER_STAT_COUNT
} TemperSchedulerStat;

uint64_t temper_scheduler_stat(TemperSchedulerStat stat);

// Observability
void temper_scheduler_dump_state(void);
bool temper_scheduler_validate(void);

#endif
