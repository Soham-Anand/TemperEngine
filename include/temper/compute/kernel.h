#ifndef TEMPER_KERNEL_H
#define TEMPER_KERNEL_H

#include "temper/core/device.h"
#include <stddef.h>
#include <stdint.h>

// Kernel registry (compute dispatch): the API expresses WHAT to compute
// (TemperKernelType), the scheduler decides WHERE (device), and each registered
// implementation (TemperKernelImpl) chooses HOW to execute on its device.
// Multiple implementations can serve one type (cpu_reference, metal_naive,
// metal_tiled, ...); registering for the same (type, device_type) replaces the
// previous implementation.

typedef struct TemperTensor TemperTensor;
struct TemperKernelImpl;

typedef enum TemperKernelType
{
    TEMPER_KERNEL_ADD = 0,
    TEMPER_KERNEL_SUB,
    TEMPER_KERNEL_MUL,
    TEMPER_KERNEL_DIV,
    TEMPER_KERNEL_RELU,
    TEMPER_KERNEL_GELU,
    TEMPER_KERNEL_SILU,
    TEMPER_KERNEL_MATMUL,
    TEMPER_KERNEL_TYPE_COUNT // append-only: fused ops, attention, layernorm, softmax...
} TemperKernelType;

// Dispatch contract: `output` must be zero-initialized before the call. The
// kernel implementation creates the output tensor (on its device) and assigns it.
// The caller owns the output resource afterwards (destroy with temper_tensor_destroy).
typedef struct TemperDispatchArgs
{
    const TemperTensor *const *inputs;
    uint32_t input_count;
    TemperTensor *output;
    const void *params; // per-kernel scalars/flags (alpha, transpose, ...)
    size_t params_size;
    void *workspace; // scratch space for high-performance kernels
    size_t workspace_size;
} TemperDispatchArgs;

typedef struct TemperKernelReport
{
    TemperKernelType type;
    uint64_t time_us;        // 0 => dispatch measured wall time is used
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t flops;
    uint8_t occupancy;       // 0 = unknown until GPU counter samples land
} TemperKernelReport;

typedef int (*TemperKernelLaunchFn)(const struct TemperKernelImpl *impl,
                                    const TemperDispatchArgs *args,
                                    TemperKernelReport *report);

typedef struct TemperKernelImpl
{
    const char *name;
    TemperKernelType type;
    TemperDeviceType device_type;
    TemperKernelLaunchFn launch;
} TemperKernelImpl;

// Per-type telemetry: launch_count + totals make average/total/hottest cheap.
// This table is the seed of the measurement-driven scheduler loop (telemetry ->
// placement); Phase 4 only collects, it does not decide.
typedef struct TemperKernelStats
{
    TemperKernelType type;
    uint64_t launch_count;
    uint64_t total_time_us;
    uint64_t total_flops;
    uint64_t total_bytes_read;
    uint64_t total_bytes_written;
} TemperKernelStats;

const char *temper_kernel_type_name(TemperKernelType type);

void temper_kernel_init(void);
int temper_kernel_register(const TemperKernelImpl *impl);
uint32_t temper_kernel_impl_count(TemperKernelType type);
const TemperKernelImpl *temper_kernel_select(TemperKernelType type, TemperDeviceType device_type);

int temper_dispatch_kernel(TemperKernelType type, const TemperDispatchArgs *args,
                           TemperKernelReport *report);

const TemperKernelStats *temper_kernel_stats(TemperKernelType type);
void temper_kernel_stats_reset(void);

#endif
