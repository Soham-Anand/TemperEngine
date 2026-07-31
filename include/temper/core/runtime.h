#ifndef TEMPER_RUNTIME_H
#define TEMPER_RUNTIME_H

#include "temper/core/device.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TEMPER_MAX_RUNTIMES 8

// ADR-003: A runtime owns one device's memory policy, synchronization, and
// lifecycle. The core only ever sees Allocate -> Pointer -> Size; it never knows
// whether memory came from calloc, mmap, Metal shared memory, or CUDA pinned
// memory. Compute dispatch lives in the kernel registry (temper/compute/kernel.h),
// not in the runtime, so this struct has no dispatch entry point.
typedef struct TemperRuntime
{
    const char *name;
    TemperDevice device;

    // Device-local memory (NULL/0-byte ops for runtimes without device-local storage)
    void *(*alloc)(size_t bytes);
    void (*free)(void *ptr);

    // CPU-accessible memory (calloc, mmap, Metal unified/shared buffer)
    void *(*alloc_host)(size_t bytes);
    void (*free_host)(void *ptr);

    int (*init)(void);
    void (*shutdown)(void);
    void (*synchronize)(void); // no-op on CPU
    void (*wait_idle)(void);
} TemperRuntime;

void temper_runtime_table_init(void);
int temper_runtime_register(TemperRuntime *runtime);
uint32_t temper_runtime_count(void);
TemperRuntime *temper_get_runtime(TemperDevice device);
TemperRuntime *temper_get_runtime_by_type(TemperDeviceType type);
TemperRuntime *temper_cpu_runtime(void);
void temper_runtime_shutdown_all(void);

#endif
