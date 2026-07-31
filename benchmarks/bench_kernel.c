#include "temper/temper.h"
#include "temper/metal/metal.h"
#include "temper/compute/kernel.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include "temper/core/platform.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// Kernel benchmark harness (Phase 4 Slice 4b). Measurement tool — not part of
// ctest. Every kernel change gets objective numbers: time, speedup, GFLOPS,
// bandwidth. The CPU column is skipped above a FLOP budget because the
// reference matmul is a naive triple loop.

static int g_metal = 0;

static int64_t flops_for(int64_t M, int64_t K, int64_t N)
{
    return 2 * M * K * N;
}

static int repeat_for(int64_t flops)
{
    if (flops < 1000000)
    {
        return 50;
    }
    if (flops < 100000000)
    {
        return 20;
    }
    if (flops < 5000000000LL)
    {
        return 5;
    }
    return 3;
}

static void fill_lin(float *d, int64_t n, float base)
{
    for (int64_t i = 0; i < n; i++)
    {
        d[i] = base + (float)i;
    }
}

// Returns best time in milliseconds, or -1 when skipped.
static double bench_cpu_matmul(int64_t M, int64_t K, int64_t N)
{
    if (2.0 * M * K * N > 8e9)
    {
        return -1.0;
    }
    int repeat = repeat_for(flops_for(M, K, N));
    TemperShape sa = temper_shape_2d(M, K);
    TemperShape sb = temper_shape_2d(K, N);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    fill_lin(a.resource->host_ptr, M * K, 0.5f);
    fill_lin(b.resource->host_ptr, K * N, 0.25f);

    double best = 1e18;
    for (int r = 0; r < repeat; r++)
    {
        uint64_t t0 = temper_time_us();
        TemperTensor out = temper_tensor_matmul(&a, &b);
        uint64_t t1 = temper_time_us();
        double us = (double)(t1 - t0);
        if (us < best)
        {
            best = us;
        }
        temper_tensor_destroy(&out);
    }
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&a);
    return best / 1000.0;
}

static double bench_gpu_matmul(int64_t M, int64_t K, int64_t N)
{
    if (!g_metal)
    {
        return -1.0;
    }
    int repeat = repeat_for(flops_for(M, K, N));
    TemperShape sa = temper_shape_2d(M, K);
    TemperShape sb = temper_shape_2d(K, N);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    fill_lin(a.resource->host_ptr, M * K, 0.5f);
    fill_lin(b.resource->host_ptr, K * N, 0.25f);
    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gb = temper_tensor_to(&b, TEMPER_DEVICE_GPU_0);

    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gb};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;

    // Warmup (compiles pipeline, touches caches).
    for (int i = 0; i < 2; i++)
    {
        memset(&gout, 0, sizeof(gout));
        temper_dispatch_kernel(TEMPER_KERNEL_MATMUL, &args, &report);
        temper_tensor_destroy(&gout);
    }

    double best = 1e18;
    for (int r = 0; r < repeat; r++)
    {
        memset(&gout, 0, sizeof(gout));
        uint64_t t0 = temper_time_us();
        temper_dispatch_kernel(TEMPER_KERNEL_MATMUL, &args, &report);
        uint64_t t1 = temper_time_us();
        double us = (double)(t1 - t0);
        if (us < best)
        {
            best = us;
        }
        temper_tensor_destroy(&gout);
    }

    temper_tensor_destroy(&gb);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&a);
    return best / 1000.0;
}

static double bench_cpu_add(int64_t n)
{
    int repeat = 20;
    TemperTensor a = temper_tensor_create(temper_shape_1d(n), TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(temper_shape_1d(n), TEMPER_DTYPE_F32);
    fill_lin(a.resource->host_ptr, n, 0.0f);
    fill_lin(b.resource->host_ptr, n, 1.0f);

    double best = 1e18;
    for (int r = 0; r < repeat; r++)
    {
        uint64_t t0 = temper_time_us();
        TemperTensor out = temper_tensor_add(&a, &b);
        uint64_t t1 = temper_time_us();
        double us = (double)(t1 - t0);
        if (us < best)
        {
            best = us;
        }
        temper_tensor_destroy(&out);
    }
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&a);
    return best / 1000.0;
}

static double bench_gpu_add(int64_t n)
{
    if (!g_metal)
    {
        return -1.0;
    }
    int repeat = 20;
    TemperTensor a = temper_tensor_create(temper_shape_1d(n), TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(temper_shape_1d(n), TEMPER_DTYPE_F32);
    fill_lin(a.resource->host_ptr, n, 0.0f);
    fill_lin(b.resource->host_ptr, n, 1.0f);
    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gb = temper_tensor_to(&b, TEMPER_DEVICE_GPU_0);

    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gb};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;
    for (int i = 0; i < 2; i++)
    {
        memset(&gout, 0, sizeof(gout));
        temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report);
        temper_tensor_destroy(&gout);
    }

    double best = 1e18;
    for (int r = 0; r < repeat; r++)
    {
        memset(&gout, 0, sizeof(gout));
        uint64_t t0 = temper_time_us();
        temper_dispatch_kernel(TEMPER_KERNEL_ADD, &args, &report);
        uint64_t t1 = temper_time_us();
        double us = (double)(t1 - t0);
        if (us < best)
        {
            best = us;
        }
        temper_tensor_destroy(&gout);
    }

    temper_tensor_destroy(&gb);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&a);
    return best / 1000.0;
}

static void bench_matmul_case(int64_t M, int64_t K, int64_t N)
{
    double cpu_ms = bench_cpu_matmul(M, K, N);
    double gpu_ms = bench_gpu_matmul(M, K, N);
    int64_t flops = flops_for(M, K, N);
    double bytes = (double)(M * K + K * N + M * N) * 4.0;

    printf("  %5lld x %5lld x %5lld | ", (long long)M, (long long)K, (long long)N);
    if (cpu_ms < 0)
        printf("%9s |", "n/a");
    else
        printf("%9.2f |", cpu_ms);
    if (gpu_ms < 0)
        printf(" %9s | %7s | %9s | %7s\n", "n/a", "n/a", "n/a", "n/a");
    else
    {
        printf(" %9.2f | %6.2fx | %8.2f | %8.2f\n", gpu_ms,
               cpu_ms < 0 ? 0.0 : cpu_ms / gpu_ms,
               (double)flops / (gpu_ms * 1e6), bytes / (gpu_ms * 1e6));
    }
}

static void bench_add_case(int64_t n)
{
    double cpu_ms = bench_cpu_add(n);
    double gpu_ms = bench_gpu_add(n);
    double bytes = (double)(3 * n) * 4.0;
    printf("  %8lld elements    | ", (long long)n);
    if (gpu_ms < 0)
        printf(" cpu %8.2f ms | gpu n/a\n", cpu_ms);
    else
    {
        double bw = bytes / (gpu_ms * 1e6);
        printf(" cpu %8.2f ms | gpu %8.2f ms | %6.2fx | gpu %8.2f GB/s\n",
               cpu_ms, gpu_ms, cpu_ms / gpu_ms, bw);
    }
}

int main(void)
{
    temper_init();
    g_metal = temper_metal_is_available();

    printf("TemperEngine kernel benchmarks\n");
    printf("Metal: %s\n\n", g_metal ? "available" : "not available");

    printf("matmul (ms | speedup | GFLOPS | GB/s)\n");
    printf("  %-19s| %9s | %9s | %6s | %9s | %8s\n",
           "M x K x N", "cpu ms", "gpu ms", "speedup", "gflops", "gb/s");
    const int64_t squares[][3] = {
        {16, 16, 16}, {64, 64, 64}, {256, 256, 256},
        {1024, 1024, 1024}, {4096, 4096, 4096},
    };
    for (size_t i = 0; i < sizeof(squares) / sizeof(squares[0]); i++)
    {
        bench_matmul_case(squares[i][0], squares[i][1], squares[i][2]);
    }
    const int64_t non_square[][3] = {
        {128, 768, 3072},
        {1024, 4096, 1024},
        {512, 64, 2048},
    };
    for (size_t i = 0; i < sizeof(non_square) / sizeof(non_square[0]); i++)
    {
        bench_matmul_case(non_square[i][0], non_square[i][1], non_square[i][2]);
    }

    printf("\nelement-wise add\n");
    bench_add_case(1 << 20);
    bench_add_case(1 << 24);

    temper_shutdown();
    return 0;
}
