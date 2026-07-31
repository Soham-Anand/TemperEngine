#include "temper/compute/kernel.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include "temper/utils/assert.h"
#include <math.h>
#include <string.h>

#define TEMPER_MAX_KERNEL_IMPLS 32

static const TemperKernelImpl *g_impls[TEMPER_MAX_KERNEL_IMPLS];
static uint32_t g_impl_count = 0;
static bool g_initialized = false;

static TemperKernelStats g_stats[TEMPER_KERNEL_TYPE_COUNT];

const char *temper_kernel_type_name(TemperKernelType type)
{
    switch (type)
    {
    case TEMPER_KERNEL_ADD:
        return "add";
    case TEMPER_KERNEL_SUB:
        return "sub";
    case TEMPER_KERNEL_MUL:
        return "mul";
    case TEMPER_KERNEL_DIV:
        return "div";
    case TEMPER_KERNEL_RELU:
        return "relu";
    case TEMPER_KERNEL_GELU:
        return "gelu";
    case TEMPER_KERNEL_SILU:
        return "silu";
    case TEMPER_KERNEL_MATMUL:
        return "matmul";
    default:
        return "unknown";
    }
}

// --- CPU reference implementations (wrap the existing tensor ops) -------------

static int cpu_binary_launch(const TemperKernelImpl *impl, const TemperDispatchArgs *args,
                             TemperKernelReport *report)
{
    (void)report;
    if (!args || args->input_count != 2 || !args->inputs || !args->output)
    {
        return -1;
    }
    switch (impl->type)
    {
    case TEMPER_KERNEL_ADD:
        *args->output = temper_tensor_add(args->inputs[0], args->inputs[1]);
        break;
    case TEMPER_KERNEL_SUB:
        *args->output = temper_tensor_sub(args->inputs[0], args->inputs[1]);
        break;
    case TEMPER_KERNEL_MUL:
        *args->output = temper_tensor_mul(args->inputs[0], args->inputs[1]);
        break;
    case TEMPER_KERNEL_DIV:
        *args->output = temper_tensor_div(args->inputs[0], args->inputs[1]);
        break;
    default:
        return -1;
    }
    return 0;
}

static int cpu_unary_launch(const TemperKernelImpl *impl, const TemperDispatchArgs *args,
                            TemperKernelReport *report)
{
    (void)impl;
    (void)report;
    if (!args || args->input_count != 1 || !args->inputs || !args->output)
    {
        return -1;
    }
    // relu/gelu/silu have no standalone tensor op yet; fall through to the
    // element-wise reference via mul on a scalar mask is overkill. Keep it
    // explicit: these are implemented directly.
    const TemperTensor *x = args->inputs[0];
    TemperShape out_shape = x->shape;
    *args->output = temper_tensor_create(out_shape, x->dtype);
    float *dst = temper_tensor_data(args->output);
    float *src = temper_tensor_data(x);
    if (!dst || !src)
    {
        return -1;
    }
    size_t n = temper_shape_count(&out_shape);
    for (size_t i = 0; i < n; i++)
    {
        float v = src[i];
        switch (impl->type)
        {
        case TEMPER_KERNEL_RELU:
            dst[i] = v > 0.0f ? v : 0.0f;
            break;
        case TEMPER_KERNEL_GELU:
            dst[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            break;
        case TEMPER_KERNEL_SILU:
            dst[i] = v / (1.0f + expf(-v));
            break;
        default:
            return -1;
        }
    }
    return 0;
}

static int cpu_matmul_launch(const TemperKernelImpl *impl, const TemperDispatchArgs *args,
                             TemperKernelReport *report)
{
    (void)impl;
    (void)report;
    if (!args || args->input_count != 2 || !args->inputs || !args->output)
    {
        return -1;
    }
    *args->output = temper_tensor_matmul(args->inputs[0], args->inputs[1]);
    return 0;
}

static TemperKernelImpl g_cpu_impls[TEMPER_KERNEL_TYPE_COUNT];

static void register_cpu_reference(TemperKernelType type, const char *name, TemperKernelLaunchFn fn)
{
    static uint32_t idx = 0;
    TemperKernelImpl *impl = &g_cpu_impls[idx++];
    impl->name = name;
    impl->type = type;
    impl->device_type = TEMPER_DEVICE_CPU;
    impl->launch = fn;
    if (temper_kernel_register(impl) != 0)
    {
        temper_error("Failed to register %s", name);
    }
}

void temper_kernel_init(void)
{
    if (g_initialized)
    {
        return;
    }
    g_initialized = true;
    memset(g_impls, 0, sizeof(g_impls));
    g_impl_count = 0;
    memset(g_stats, 0, sizeof(g_stats));

    register_cpu_reference(TEMPER_KERNEL_ADD, "cpu_add", cpu_binary_launch);
    register_cpu_reference(TEMPER_KERNEL_SUB, "cpu_sub", cpu_binary_launch);
    register_cpu_reference(TEMPER_KERNEL_MUL, "cpu_mul", cpu_binary_launch);
    register_cpu_reference(TEMPER_KERNEL_DIV, "cpu_div", cpu_binary_launch);
    register_cpu_reference(TEMPER_KERNEL_RELU, "cpu_relu", cpu_unary_launch);
    register_cpu_reference(TEMPER_KERNEL_GELU, "cpu_gelu", cpu_unary_launch);
    register_cpu_reference(TEMPER_KERNEL_SILU, "cpu_silu", cpu_unary_launch);
    register_cpu_reference(TEMPER_KERNEL_MATMUL, "cpu_matmul", cpu_matmul_launch);

    temper_info("Kernel table initialized: %u implementations", g_impl_count);
}

int temper_kernel_register(const TemperKernelImpl *impl)
{
    if (!impl || !impl->name || !impl->launch)
    {
        temper_error("Failed to register kernel: invalid implementation");
        return -1;
    }
    temper_kernel_init();

    for (uint32_t i = 0; i < g_impl_count; i++)
    {
        if (g_impls[i]->type == impl->type && g_impls[i]->device_type == impl->device_type)
        {
            temper_info("Replaced %s with %s (%s)", g_impls[i]->name, impl->name,
                        temper_kernel_type_name(impl->type));
            g_impls[i] = impl;
            return 0;
        }
    }
    if (g_impl_count >= TEMPER_MAX_KERNEL_IMPLS)
    {
        temper_error("Failed to register %s: kernel table full", impl->name);
        return -1;
    }
    g_impls[g_impl_count++] = impl;
    temper_info("Registered kernel %s (%s)", impl->name,
                temper_kernel_type_name(impl->type));
    return 0;
}

uint32_t temper_kernel_impl_count(TemperKernelType type)
{
    temper_kernel_init();
    uint32_t count = 0;
    for (uint32_t i = 0; i < g_impl_count; i++)
    {
        if (g_impls[i]->type == type)
        {
            count++;
        }
    }
    return count;
}

const TemperKernelImpl *temper_kernel_select(TemperKernelType type, TemperDeviceType device_type)
{
    temper_kernel_init();
    for (uint32_t i = 0; i < g_impl_count; i++)
    {
        if (g_impls[i]->type == type && g_impls[i]->device_type == device_type)
        {
            return g_impls[i];
        }
    }
    return NULL;
}

static TemperDeviceType dispatch_device(const TemperDispatchArgs *args)
{
    if (args && args->inputs && args->input_count > 0 && args->inputs[0])
    {
        const TemperTensor *t = args->inputs[0];
        if (t->resource)
        {
            return t->resource->device.type;
        }
    }
    return TEMPER_DEVICE_CPU;
}

static void kernel_cost(TemperKernelType type, const TemperDispatchArgs *args,
                        uint64_t *flops, uint64_t *bytes_read, uint64_t *bytes_written)
{
    uint64_t fl = 0;
    uint64_t rb = 0;
    uint64_t wb = 0;

    for (uint32_t i = 0; i < args->input_count; i++)
    {
        if (args->inputs[i])
        {
            rb += (uint64_t)temper_tensor_bytes(args->inputs[i]);
        }
    }
    if (args->output)
    {
        wb = (uint64_t)temper_tensor_bytes(args->output);
        size_t n = temper_shape_count(&args->output->shape);
        if (type == TEMPER_KERNEL_MATMUL)
        {
            int64_t m = args->output->shape.dims[0];
            int64_t nn = args->output->shape.ndim > 1 ? args->output->shape.dims[1] : 1;
            int64_t k = args->input_count > 0 && args->inputs[0] ? args->inputs[0]->shape.dims[1] : 0;
            fl = (uint64_t)(2 * m * nn * k);
        }
        else
        {
            fl = (uint64_t)n; // element-wise: one operation per element
        }
    }

    *flops = fl;
    *bytes_read = rb;
    *bytes_written = wb;
}

int temper_dispatch_kernel(TemperKernelType type, const TemperDispatchArgs *args,
                           TemperKernelReport *report)
{
    TEMPER_ASSERT_MSG(args != NULL, "Dispatch args required");
    TEMPER_ASSERT_MSG(report != NULL, "Kernel report required");
    if (!args || !report)
    {
        return -1;
    }
    TEMPER_ASSERT_MSG(args->output != NULL, "Dispatch output required");

    temper_kernel_init();
    memset(report, 0, sizeof(*report));

    TemperDeviceType dev = dispatch_device(args);
    const TemperKernelImpl *impl = temper_kernel_select(type, dev);
    if (!impl)
    {
        temper_error("No kernel implementation for %s on device type %d",
                     temper_kernel_type_name(type), (int)dev);
        return -1;
    }

    uint64_t t0 = temper_time_us();
    int rc = impl->launch(impl, args, report);
    uint64_t t1 = temper_time_us();
    if (rc != 0)
    {
        temper_error("Kernel %s failed (%d)", impl->name, rc);
        return rc;
    }

    report->type = type;
    report->time_us = report->time_us == 0 ? (t1 - t0) : report->time_us;
    kernel_cost(type, args, &report->flops, &report->bytes_read, &report->bytes_written);

    TemperKernelStats *st = &g_stats[type];
    st->launch_count++;
    st->total_time_us += report->time_us;
    st->total_flops += report->flops;
    st->total_bytes_read += report->bytes_read;
    st->total_bytes_written += report->bytes_written;
    return 0;
}

const TemperKernelStats *temper_kernel_stats(TemperKernelType type)
{
    temper_kernel_init();
    return &g_stats[type];
}

void temper_kernel_stats_reset(void)
{
    temper_kernel_init();
    memset(g_stats, 0, sizeof(g_stats));
}
