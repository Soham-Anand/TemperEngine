#ifndef TEMPER_RESOURCE_H
#define TEMPER_RESOURCE_H

#include "temper/core/device.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

struct TemperGraphNode;

// Physical memory tiers (ADR-004 Amendment 1).
// "Recomputable" is NOT a tier — it is a resource state flag.
typedef enum TemperMemTier
{
    TEMPER_TIER_GPU = 0,
    TEMPER_TIER_CPU,
    TEMPER_TIER_COMPRESSED,
    TEMPER_TIER_SSD,
    TEMPER_TIER_COUNT
} TemperMemTier;

// Resource state flags (packed, one cache-line fetch covers all state).
#define TEMPER_RESOURCE_RESIDENT     (1u << 0) // data present in memory (host buffer or compressed blob)
#define TEMPER_RESOURCE_COMPRESSED   (1u << 1) // data stored as backend-owned blob (implies RESIDENT)
#define TEMPER_RESOURCE_RECOMPUTABLE (1u << 2) // can be regenerated from origin op
#define TEMPER_RESOURCE_PINNED       (1u << 3) // scheduler cannot evict

typedef struct TemperResource
{
    uint32_t id;                    // Unique resource ID for tracking
    TemperDevice device;            // Device where memory lives
    void *native;                   // Backend handle (MTLBuffer, cudaMalloc, raw ptr)
    float *host_ptr;                // Host CPU pointer (NULL if GPU only or non-resident)
    size_t bytes;                   // Uncompressed size in bytes
    uint64_t last_access;           // Microsecond timestamp of last access
    uint32_t refcount;              // Reference count
    uint32_t flags;                 // TEMPER_RESOURCE_* flags
    TemperMemTier tier;             // Physical tier this resource is tracked in
    uint32_t access_count;          // Lifetime access counter (deterministic scoring)
    uint64_t created_at;            // Microsecond timestamp of creation
    struct TemperGraphNode *origin; // Graph node that created this resource (recompute source)
    void *compressed_blob;          // Backend-owned compression metadata + data (opaque)
    size_t compressed_size;         // Compressed footprint in bytes
} TemperResource;

TemperResource *temper_resource_create(TemperDevice device, size_t bytes);
void temper_resource_retain(TemperResource *res);
void temper_resource_release(TemperResource *res);
int temper_resource_migrate(TemperResource *res, TemperDevice target_device);
void temper_resource_touch(TemperResource *res);

// State queries
bool temper_resource_is_resident(const TemperResource *res);
bool temper_resource_is_compressed(const TemperResource *res);
bool temper_resource_is_recomputable(const TemperResource *res);
bool temper_resource_is_pinned(const TemperResource *res);

#endif
