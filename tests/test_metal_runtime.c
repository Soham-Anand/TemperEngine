#include "temper/temper.h"
#include "temper/core/device.h"
#include "temper/core/resource.h"
#include "temper/core/runtime.h"
#include "temper/compute/kernel.h"
#include "temper/memory/scheduler.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__APPLE__) && defined(TEMPER_HAS_METAL)
#define TEMPER_METAL_TESTS 1
#else
#define TEMPER_METAL_TESTS 0
#endif

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

static TemperTensor make_f32(int64_t n)
{
    TemperShape s = temper_shape_1d(n);
    return temper_tensor_create(s, TEMPER_DTYPE_F32);
}

#if TEMPER_METAL_TESTS

#include "temper/metal/metal.h"

TEST(metal_is_available)
{
    ASSERT(temper_metal_is_available());
}

TEST(lazy_init_on_first_gpu_resource)
{
    // No GPU runtime until something asks for the GPU device.
    ASSERT(temper_get_runtime(TEMPER_DEVICE_GPU_0) == NULL);

    TemperShape s = temper_shape_1d(8);
    TemperTensor g = temper_tensor_create_on_device(s, TEMPER_DTYPE_F32, TEMPER_DEVICE_GPU_0);
    ASSERT(g.resource != NULL);
    ASSERT(temper_get_runtime(TEMPER_DEVICE_GPU_0) != NULL);
    ASSERT(strcmp(g.resource->allocator->name, "metal") == 0);
    ASSERT(g.resource->tier == TEMPER_TIER_GPU);
    ASSERT(g.resource->device.type == TEMPER_DEVICE_GPU);

    // Caps registered for GPU_0.
    TemperDeviceCaps caps = temper_device_get_caps(TEMPER_DEVICE_GPU_0);
    ASSERT(caps.total_memory > 0);
    ASSERT(caps.supports_fp16);

    temper_tensor_destroy(&g);
}

TEST(metal_impls_registered)
{
    const TemperKernelImpl *add = temper_kernel_select(TEMPER_KERNEL_ADD, TEMPER_DEVICE_GPU);
    ASSERT(add != NULL);
    ASSERT(strcmp(add->name, "metal_add") == 0);
    ASSERT(strcmp(temper_kernel_select(TEMPER_KERNEL_MATMUL, TEMPER_DEVICE_GPU)->name,
                  "metal_matmul_tiled") == 0);
    ASSERT(temper_kernel_select(TEMPER_KERNEL_GELU, TEMPER_DEVICE_GPU) != NULL);
}

TEST(gpu_tensor_data_via_shared_memory)
{
    TemperShape s = temper_shape_1d(16);
    TemperTensor g = temper_tensor_create_on_device(s, TEMPER_DTYPE_F32, TEMPER_DEVICE_GPU_0);
    float *data = temper_tensor_data(&g);
    ASSERT(data != NULL);
    for (int i = 0; i < 16; i++)
    {
        data[i] = (float)i;
    }
    for (int i = 0; i < 16; i++)
    {
        ASSERT(data[i] == (float)i);
    }
    temper_tensor_destroy(&g);
}

TEST(gpu_add_matches_cpu)
{
    TemperTensor a = make_f32(64);
    TemperTensor b = make_f32(64);
    for (int i = 0; i < 64; i++)
    {
        a.resource->host_ptr[i] = (float)i;
        b.resource->host_ptr[i] = (float)(i * 2);
    }

    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gb = temper_tensor_to(&b, TEMPER_DEVICE_GPU_0);

    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gb};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report) == 0);
    ASSERT(temper_tensor_device(&gout).type == TEMPER_DEVICE_GPU);
    ASSERT(report.type == TEMPER_KERNEL_ADD);
    ASSERT(report.flops == 64);
    ASSERT(report.bytes_read == 64 * 4 * 2);
    ASSERT(report.bytes_written == 64 * 4);

    float *o = temper_tensor_data(&gout);
    for (int i = 0; i < 64; i++)
    {
        ASSERT(o[i] == (float)(i * 3));
    }

    temper_tensor_destroy(&gout);
    temper_tensor_destroy(&gb);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(gpu_unary_activations)
{
    TemperTensor x = make_f32(5);
    float vals[5] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    memcpy(x.resource->host_ptr, vals, sizeof(vals));
    TemperTensor gx = temper_tensor_to(&x, TEMPER_DEVICE_GPU_0);

    const TemperTensor *inputs[1] = {&gx};
    TemperTensor gout;
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 1, .output = &gout};
    TemperKernelReport report;

    memset(&gout, 0, sizeof(gout));
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_RELU, &args, &report) == 0);
    float *o = temper_tensor_data(&gout);
    ASSERT(o[0] == 0.0f && o[3] == 1.0f && o[4] == 3.0f);
    temper_tensor_destroy(&gout);

    memset(&gout, 0, sizeof(gout));
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_SILU, &args, &report) == 0);
    o = temper_tensor_data(&gout);
    ASSERT(fabsf(o[3] - 0.7310586f) < 1e-4f && fabsf(o[2]) < 1e-6f);
    temper_tensor_destroy(&gout);

    memset(&gout, 0, sizeof(gout));
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_GELU, &args, &report) == 0);
    o = temper_tensor_data(&gout);
    ASSERT(fabsf(o[4] - 3.0f) < 1e-2f && o[2] == 0.0f);
    temper_tensor_destroy(&gout);

    temper_tensor_destroy(&gx);
    temper_tensor_destroy(&x);
}

TEST(gpu_matmul_matches_cpu)
{
    TemperShape sa = temper_shape_2d(4, 3);
    TemperShape sb = temper_shape_2d(3, 2);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    for (int i = 0; i < 12; i++)
    {
        a.resource->host_ptr[i] = (float)i;
    }
    for (int i = 0; i < 6; i++)
    {
        b.resource->host_ptr[i] = (float)(i + 1);
    }

    TemperTensor ref = temper_tensor_matmul(&a, &b);
    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gb = temper_tensor_to(&b, TEMPER_DEVICE_GPU_0);

    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gb};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_MATMUL, &args, &report) == 0);
    ASSERT(temper_tensor_device(&gout).type == TEMPER_DEVICE_GPU);
    ASSERT(report.flops == 2 * 4 * 2 * 3);

    float *o = temper_tensor_data(&gout);
    float *r = ref.resource->host_ptr;
    ASSERT(arrays_close(o, r, 8, 1e-4f));

    temper_tensor_destroy(&gout);
    temper_tensor_destroy(&gb);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&ref);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(zero_copy_to_cpu_migrates)
{
    TemperShape s = temper_shape_1d(16);
    TemperTensor cpu = make_f32(16);
    for (int i = 0; i < 16; i++)
    {
        cpu.resource->host_ptr[i] = (float)i;
    }

    TemperTensor g = temper_tensor_to(&cpu, TEMPER_DEVICE_GPU_0);
    ASSERT(temper_tensor_device(&g).type == TEMPER_DEVICE_GPU);

    // GPU -> CPU must be a metadata-only migrate (no copy).
    TemperTensor back = temper_tensor_to(&g, TEMPER_DEVICE_CPU_0);
    ASSERT(temper_tensor_device(&back).type == TEMPER_DEVICE_CPU);
    ASSERT(back.resource == g.resource);
    ASSERT(back.resource->tier == TEMPER_TIER_CPU);
    ASSERT(temper_tensor_device(&g).type == TEMPER_DEVICE_CPU);

    float *data = temper_tensor_data(&back);
    for (int i = 0; i < 16; i++)
    {
        ASSERT(data[i] == (float)i);
    }

    temper_tensor_destroy(&back);
    temper_tensor_destroy(&g);
    temper_tensor_destroy(&cpu);
}

TEST(gpu_broadcast_rejected)
{
    TemperTensor a = make_f32(4);
    TemperTensor s = make_f32(1);
    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gs = temper_tensor_to(&s, TEMPER_DEVICE_GPU_0);

    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gs};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;
    // Equal-shape requirement is documented for Phase 4; mismatch must fail
    // cleanly rather than produce garbage.
    ASSERT(temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report) == -1);

    temper_tensor_destroy(&gs);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&s);
    temper_tensor_destroy(&a);
}

#endif // TEMPER_METAL_TESTS

int main(void)
{
    printf("test_metal_runtime\n");
#if TEMPER_METAL_TESTS
    if (!temper_metal_is_available())
    {
        printf("  Metal unavailable — skipping\n");
        return 0;
    }
    temper_log_set_level(TEMPER_LOG_WARN);
    RUN(metal_is_available);
    RUN(lazy_init_on_first_gpu_resource);
    RUN(metal_impls_registered);
    RUN(gpu_tensor_data_via_shared_memory);
    RUN(gpu_add_matches_cpu);
    RUN(gpu_unary_activations);
    RUN(gpu_matmul_matches_cpu);
    RUN(zero_copy_to_cpu_migrates);
    RUN(gpu_broadcast_rejected);
#else
    printf("  skipped (requires macOS + Metal)\n");
    tests_run = 1;
    tests_passed = 1;
#endif
    printf("  %d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
