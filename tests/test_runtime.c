#include "temper/temper.h"
#include "temper/core/device.h"
#include "temper/core/resource.h"
#include "temper/core/runtime.h"
#include "temper/compute/kernel.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;
static int test_failed = 0;

#define TEST(name) static void name(void)
#define RUN(name)                                                             \
    do                                                                        \
    {                                                                         \
        tests_run++;                                                          \
        test_failed = 0;                                                      \
        printf("  %-40s", #name);                                             \
        name();                                                               \
        if (test_failed)                                                      \
            printf("FAIL\n");                                                 \
        else                                                                  \
        {                                                                     \
            tests_passed++;                                                   \
            printf("PASS\n");                                                 \
        }                                                                     \
    } while (0)

#define ASSERT(cond)                                                          \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            printf("\n    %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            test_failed = 1;                                                  \
            return;                                                           \
        }                                                                     \
    } while (0)

static TemperTensor make_f32(int64_t n)
{
    TemperShape s = temper_shape_1d(n);
    return temper_tensor_create(s, TEMPER_DTYPE_F32);
}

static void fill_lin(float *data, int64_t n, float base)
{
    for (int64_t i = 0; i < n; i++)
    {
        data[i] = base + (float)i;
    }
}

static int arrays_close(const float *a, const float *b, int64_t n, float eps)
{
    for (int64_t i = 0; i < n; i++)
    {
        if (fabsf(a[i] - b[i]) > eps)
        {
            return 0;
        }
    }
    return 1;
}

// --- custom implementations for registry tests --------------------------------

static const TemperKernelImpl *s_real_matmul;
static TemperKernelImpl g_test_matmul;

static int delegate_matmul_launch(const TemperKernelImpl *impl,
                                  const TemperDispatchArgs *args,
                                  TemperKernelReport *report)
{
    (void)impl;
    return s_real_matmul->launch(s_real_matmul, args, report);
}

static const TemperKernelImpl *s_real_mul;
static TemperKernelImpl g_counting_mul;
static void *s_seen_workspace = NULL;
static size_t s_seen_workspace_size = 0;
static uint32_t s_wrapper_calls = 0;

static int counting_mul_launch(const TemperKernelImpl *impl,
                               const TemperDispatchArgs *args,
                               TemperKernelReport *report)
{
    (void)impl;
    s_wrapper_calls++;
    s_seen_workspace = args->workspace;
    s_seen_workspace_size = args->workspace_size;
    return s_real_mul->launch(s_real_mul, args, report);
}

// --- tests ---------------------------------------------------------------------

TEST(cpu_runtime_registered)
{
    temper_log_set_level(TEMPER_LOG_WARN);
    TemperRuntime *cpu = temper_get_runtime(TEMPER_DEVICE_CPU_0);
    ASSERT(cpu != NULL);
    ASSERT(strcmp(cpu->name, "cpu") == 0);
    ASSERT(cpu->alloc_host != NULL && cpu->free_host != NULL);
    ASSERT(temper_get_runtime_by_type(TEMPER_DEVICE_CPU) == cpu);
    ASSERT(temper_cpu_runtime() == cpu);
}

TEST(unregistered_gpu_device_has_no_runtime)
{
    TemperDevice gpu5 = {TEMPER_DEVICE_GPU, 5};
    ASSERT(temper_get_runtime(gpu5) == NULL);
}

TEST(runtime_register_replaces_same_device)
{
    static TemperRuntime fake;
    fake.name = "fake_gpu";
    fake.device = (TemperDevice){TEMPER_DEVICE_GPU, 1};
    fake.alloc_host = temper_cpu_runtime()->alloc_host;
    fake.free_host = temper_cpu_runtime()->free_host;
    fake.init = temper_cpu_runtime()->init;
    fake.shutdown = temper_cpu_runtime()->shutdown;

    ASSERT(temper_runtime_register(&fake) == 0);
    TemperRuntime *got = temper_get_runtime((TemperDevice){TEMPER_DEVICE_GPU, 1});
    ASSERT(got == &fake);
    ASSERT(strcmp(got->name, "fake_gpu") == 0);

    static TemperRuntime fake2;
    fake2.name = "fake_gpu_v2";
    fake2.device = (TemperDevice){TEMPER_DEVICE_GPU, 1};
    fake2.alloc_host = temper_cpu_runtime()->alloc_host;
    fake2.free_host = temper_cpu_runtime()->free_host;
    fake2.init = temper_cpu_runtime()->init;
    fake2.shutdown = temper_cpu_runtime()->shutdown;
    ASSERT(temper_runtime_register(&fake2) == 0);
    got = temper_get_runtime((TemperDevice){TEMPER_DEVICE_GPU, 1});
    ASSERT(got == &fake2);
}

TEST(resource_alloc_routed_through_runtime)
{
    TemperResource *res = temper_resource_create(TEMPER_DEVICE_CPU_0, 1024);
    ASSERT(res != NULL);
    ASSERT(res->allocator == temper_cpu_runtime());
    ASSERT(temper_resource_is_resident(res));
    // calloc semantics preserved: zeroed
    for (size_t i = 0; i < 256; i++)
    {
        ASSERT(res->host_ptr[i] == 0.0f);
    }
    res->host_ptr[0] = 42.0f;
    temper_resource_release(res);
}

TEST(cpu_kernel_impls_present)
{
    const TemperKernelType types[] = {
        TEMPER_KERNEL_ADD, TEMPER_KERNEL_SUB, TEMPER_KERNEL_MUL, TEMPER_KERNEL_DIV,
        TEMPER_KERNEL_RELU, TEMPER_KERNEL_GELU, TEMPER_KERNEL_SILU, TEMPER_KERNEL_MATMUL,
    };
    const char *names[] = {
        "cpu_add", "cpu_sub", "cpu_mul", "cpu_div",
        "cpu_relu", "cpu_gelu", "cpu_silu", "cpu_matmul",
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
    {
        const TemperKernelImpl *impl = temper_kernel_select(types[i], TEMPER_DEVICE_CPU);
        ASSERT(impl != NULL);
        ASSERT(strcmp(impl->name, names[i]) == 0);
    }
    // No GPU implementations exist yet.
    ASSERT(temper_kernel_select(TEMPER_KERNEL_ADD, TEMPER_DEVICE_GPU) == NULL);
    ASSERT(temper_kernel_select(TEMPER_KERNEL_MATMUL, TEMPER_DEVICE_GPU) == NULL);
}

TEST(kernel_register_replaces_implementation)
{
    s_real_matmul = temper_kernel_select(TEMPER_KERNEL_MATMUL, TEMPER_DEVICE_CPU);
    ASSERT(s_real_matmul != NULL);

    g_test_matmul.name = "test_matmul_impl";
    g_test_matmul.type = TEMPER_KERNEL_MATMUL;
    g_test_matmul.device_type = TEMPER_DEVICE_CPU;
    g_test_matmul.launch = delegate_matmul_launch;

    ASSERT(temper_kernel_register(&g_test_matmul) == 0);
    const TemperKernelImpl *impl = temper_kernel_select(TEMPER_KERNEL_MATMUL, TEMPER_DEVICE_CPU);
    ASSERT(impl == &g_test_matmul);
    ASSERT(strcmp(impl->name, "test_matmul_impl") == 0);
}

TEST(dispatch_add_matches_direct)
{
    TemperTensor a = make_f32(64);
    TemperTensor b = make_f32(64);
    fill_lin(a.resource->host_ptr, 64, 1.0f);
    fill_lin(b.resource->host_ptr, 64, 10.0f);

    TemperTensor direct = temper_tensor_add(&a, &b);
    TemperTensor via_kernel;
    memset(&via_kernel, 0, sizeof(via_kernel));
    const TemperTensor *inputs[2] = {&a, &b};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &via_kernel};
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report) == 0);
    ASSERT(via_kernel.resource != NULL);

    float *d = direct.resource->host_ptr;
    float *v = via_kernel.resource->host_ptr;
    ASSERT(arrays_close(d, v, 64, 1e-4f));
    ASSERT(report.type == TEMPER_KERNEL_ADD);
    ASSERT(report.bytes_read == 64 * 4 * 2);
    ASSERT(report.bytes_written == 64 * 4);
    ASSERT(report.flops == 64);

    temper_tensor_destroy(&direct);
    temper_tensor_destroy(&via_kernel);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(dispatch_matmul_matches_direct_and_reports_cost)
{
    // 4x3 * 3x2 -> 4x2
    TemperShape sa = temper_shape_2d(4, 3);
    TemperShape sb = temper_shape_2d(3, 2);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    fill_lin(a.resource->host_ptr, 12, 0.0f);
    fill_lin(b.resource->host_ptr, 6, 0.0f);

    TemperTensor direct = temper_tensor_matmul(&a, &b);
    TemperTensor via_kernel;
    memset(&via_kernel, 0, sizeof(via_kernel));
    const TemperTensor *inputs[2] = {&a, &b};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &via_kernel};
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_MATMUL, &args, &report) == 0);

    ASSERT(arrays_close(direct.resource->host_ptr, via_kernel.resource->host_ptr, 8, 1e-4f));
    ASSERT(report.type == TEMPER_KERNEL_MATMUL);
    ASSERT(report.bytes_read == 48 + 24);
    ASSERT(report.bytes_written == 32);
    ASSERT(report.flops == 2 * 4 * 2 * 3);

    temper_tensor_destroy(&direct);
    temper_tensor_destroy(&via_kernel);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(dispatch_unary_activations)
{
    TemperTensor x = make_f32(5);
    float vals[5] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    memcpy(x.resource->host_ptr, vals, sizeof(vals));

    const TemperTensor *inp[1] = {&x};
    TemperTensor out;
    memset(&out, 0, sizeof(out));
    TemperDispatchArgs args = {.inputs = inp, .input_count = 1, .output = &out};
    TemperKernelReport report;

    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_RELU, &args, &report) == 0);
    float *o = out.resource->host_ptr;
    ASSERT(o[0] == 0.0f && o[1] == 0.0f && o[2] == 0.0f && o[3] == 1.0f && o[4] == 3.0f);
    temper_tensor_destroy(&out);
    memset(&out, 0, sizeof(out));

    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_GELU, &args, &report) == 0);
    o = out.resource->host_ptr;
    ASSERT(o[0] < 0.01f && o[2] == 0.0f && fabsf(o[4] - 3.0f) < 1e-2f);
    temper_tensor_destroy(&out);
    memset(&out, 0, sizeof(out));

    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_SILU, &args, &report) == 0);
    o = out.resource->host_ptr;
    ASSERT(fabsf(o[2]) < 1e-6f && fabsf(o[3] - 0.7310586f) < 1e-4f);

    temper_tensor_destroy(&out);
    temper_tensor_destroy(&x);
}

TEST(dispatch_fails_without_impl_for_device)
{
    // Creating a GPU tensor works (allocation falls back to the CPU runtime),
    // but no GPU kernel implementation exists -> dispatch must fail cleanly.
    TemperTensor a = make_f32(8);
    TemperTensor g = temper_tensor_create_on_device(a.shape, a.dtype, TEMPER_DEVICE_GPU_0);
    ASSERT(g.resource != NULL);

    TemperTensor out;
    memset(&out, 0, sizeof(out));
    const TemperTensor *inputs[2] = {&g, &g};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &out};
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report) == -1);

    temper_tensor_destroy(&g);
    temper_tensor_destroy(&a);
}

TEST(kernel_stats_accumulate_and_reset)
{
    temper_kernel_stats_reset();

    TemperTensor a = make_f32(16);
    TemperTensor b = make_f32(16);
    fill_lin(a.resource->host_ptr, 16, 0.0f);
    fill_lin(b.resource->host_ptr, 16, 5.0f);

    const TemperTensor *inputs[2] = {&a, &b};
    TemperTensor out;
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &out};
    TemperKernelReport report;

    for (int i = 0; i < 3; i++)
    {
        memset(&out, 0, sizeof(out));
        ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report) == 0);
        temper_tensor_destroy(&out);
    }

    const TemperKernelStats *st = temper_kernel_stats(TEMPER_KERNEL_ADD);
    ASSERT(st->launch_count == 3);
    ASSERT(st->total_bytes_read == 3 * 16 * 4 * 2);
    ASSERT(st->total_bytes_written == 3 * 16 * 4);
    ASSERT(st->total_flops == 3 * 16);

    temper_kernel_stats_reset();
    st = temper_kernel_stats(TEMPER_KERNEL_ADD);
    ASSERT(st->launch_count == 0);

    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(workspace_passes_through_to_impl)
{
    s_real_mul = temper_kernel_select(TEMPER_KERNEL_MUL, TEMPER_DEVICE_CPU);
    ASSERT(s_real_mul != NULL);

    g_counting_mul.name = "test_counting_mul";
    g_counting_mul.type = TEMPER_KERNEL_MUL;
    g_counting_mul.device_type = TEMPER_DEVICE_CPU;
    g_counting_mul.launch = counting_mul_launch;
    ASSERT(temper_kernel_register(&g_counting_mul) == 0);

    s_wrapper_calls = 0;
    s_seen_workspace = NULL;

    char scratch[512];
    TemperTensor a = make_f32(8);
    TemperTensor b = make_f32(8);
    fill_lin(a.resource->host_ptr, 8, 1.0f);
    fill_lin(b.resource->host_ptr, 8, 2.0f);

    TemperTensor out;
    memset(&out, 0, sizeof(out));
    const TemperTensor *inputs[2] = {&a, &b};
    TemperDispatchArgs args = {
        .inputs = inputs,
        .input_count = 2,
        .output = &out,
        .workspace = scratch,
        .workspace_size = sizeof(scratch),
    };
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_MUL, &args, &report) == 0);
    ASSERT(s_wrapper_calls == 1);
    ASSERT(s_seen_workspace == scratch);
    ASSERT(s_seen_workspace_size == sizeof(scratch));
    float *o = out.resource->host_ptr;
    ASSERT(o[0] == 2.0f && o[7] == 8.0f * 9.0f);

    temper_tensor_destroy(&out);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(kernel_type_names)
{
    ASSERT(strcmp(temper_kernel_type_name(TEMPER_KERNEL_ADD), "add") == 0);
    ASSERT(strcmp(temper_kernel_type_name(TEMPER_KERNEL_MATMUL), "matmul") == 0);
    ASSERT(strcmp(temper_kernel_type_name(TEMPER_KERNEL_GELU), "gelu") == 0);
}

int main(void)
{
    printf("test_runtime\n");
    RUN(cpu_runtime_registered);
    RUN(unregistered_gpu_device_has_no_runtime);
    RUN(runtime_register_replaces_same_device);
    RUN(resource_alloc_routed_through_runtime);
    RUN(cpu_kernel_impls_present);
    RUN(kernel_register_replaces_implementation);
    RUN(dispatch_add_matches_direct);
    RUN(dispatch_matmul_matches_direct_and_reports_cost);
    RUN(dispatch_unary_activations);
    RUN(dispatch_fails_without_impl_for_device);
    RUN(kernel_stats_accumulate_and_reset);
    RUN(workspace_passes_through_to_impl);
    RUN(kernel_type_names);

    printf("  %d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
