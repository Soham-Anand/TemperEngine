#ifndef TEMPER_RESOURCE_H
#define TEMPER_RESOURCE_H

#include "temper/core/device.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

struct TemperGraphNode;

typedef struct TemperResource
{
    uint32_t id;                    // Unique resource ID for tracking
    TemperDevice device;            // Device where memory lives
    void *native;                   // Backend handle (MTLBuffer, cudaMalloc, raw ptr)
    float *host_ptr;                // Host CPU pointer (NULL if GPU only)
    size_t bytes;                   // Size in bytes
    uint64_t last_access;           // Microsecond timestamp of last access
    uint32_t refcount;              // Reference count
    bool pinned;                    // Pinned flag (cannot be evicted by scheduler)
    bool recomputable;              // Can be regenerated from inputs
    struct TemperGraphNode *origin; // Graph node that created this resource
} TemperResource;

TemperResource *temper_resource_create(TemperDevice device, size_t bytes);
void temper_resource_retain(TemperResource *res);
void temper_resource_release(TemperResource *res);
int temper_resource_migrate(TemperResource *res, TemperDevice target_device);
void temper_resource_touch(TemperResource *res);

#endif
