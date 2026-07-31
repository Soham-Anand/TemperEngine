#include "temper/temper.h"
#include "temper/metal/metal.h"
#include "temper/compute/kernel.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include <stdio.h>
#include <string.h>

// Phase 4 end-to-end demo:
//   create on CPU -> temper_tensor_to(GPU) -> GPU compute -> to(CPU)
// Prints the matmul result and the kernel telemetry report.

static float get(const TemperTensor *t, int64_t i, int64_t j)
{
    float *d = temper_tensor_data(t);
    return d[i * t->shape.dims[1] + j];
}

int main(void)
{
    if (temper_init() != 0)
    {
        return 1;
    }
    if (!temper_metal_is_available())
    {
        printf("Metal backend unavailable on this machine.\n");
        temper_shutdown();
        return 1;
    }

    // 4x3 * 3x2 on the CPU.
    TemperShape sa = temper_shape_2d(4, 3);
    TemperShape sb = temper_shape_2d(3, 2);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    for (int64_t i = 0; i < 12; i++)
    {
        a.resource->host_ptr[i] = (float)(i + 1);
    }
    for (int64_t i = 0; i < 6; i++)
    {
        b.resource->host_ptr[i] = (float)(i + 1);
    }

    // CPU reference.
    TemperTensor ref = temper_tensor_matmul(&a, &b);

    // GPU compute: move inputs (GPU tensors are Metal shared memory), matmul.
    TemperTensor ga = temper_tensor_to(&a, TEMPER_DEVICE_GPU_0);
    TemperTensor gb = temper_tensor_to(&b, TEMPER_DEVICE_GPU_0);
    TemperTensor gout;
    memset(&gout, 0, sizeof(gout));
    const TemperTensor *inputs[2] = {&ga, &gb};
    TemperDispatchArgs args = {.inputs = inputs, .input_count = 2, .output = &gout};
    TemperKernelReport report;
    if (temper_dispatch_kernel(TEMPER_KERNEL_MATMUL, &args, &report) != 0)
    {
        printf("GPU matmul failed\n");
        return 1;
    }

    // Move the GPU result back to the CPU (zero-copy migrate on unified memory).
    TemperTensor result = temper_tensor_to(&gout, TEMPER_DEVICE_CPU_0);

    printf("Metal matmul result (4x3 * 3x2):\n");
    for (int64_t i = 0; i < 4; i++)
    {
        printf("  [");
        for (int64_t j = 0; j < 2; j++)
        {
            printf(" %7.2f", get(&result, i, j));
        }
        printf(" ]\n");
    }

    int ok = 1;
    for (int64_t i = 0; i < 8; i++)
    {
        if (result.resource->host_ptr[i] != ref.resource->host_ptr[i])
        {
            ok = 0;
            break;
        }
    }
    printf("Matches CPU reference: %s\n", ok ? "yes" : "NO");

    printf("Kernel report: %s  time=%llu us  flops=%llu  read=%llu B  write=%llu B\n",
           temper_kernel_type_name(report.type), report.time_us, report.flops,
           report.bytes_read, report.bytes_written);

    temper_tensor_destroy(&result);
    temper_tensor_destroy(&gout);
    temper_tensor_destroy(&gb);
    temper_tensor_destroy(&ga);
    temper_tensor_destroy(&ref);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&a);

    temper_shutdown();
    return ok ? 0 : 1;
}
