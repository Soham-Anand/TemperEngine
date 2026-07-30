#include "temper/temper.h"
#include "temper/core/device.h"
#include "temper/core/resource.h"
#include "temper/math/tensor.h"
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

#define ASSERT_FLOAT(a, b, eps) (fabsf((a) - (b)) < (eps))

TEST(test_device_table_and_caps)
{
    temper_device_table_init();
    ASSERT(temper_device_count() >= 1);
    TemperDevice cpu0 = TEMPER_DEVICE_CPU_0;
    ASSERT(temper_device_is_cpu(cpu0));
    ASSERT(!temper_device_is_gpu(cpu0));
    ASSERT(strcmp(temper_device_name(cpu0), "CPU:0") == 0);

    TemperDeviceCaps gpu_caps = {0};
    gpu_caps.supports_fp16 = true;
    gpu_caps.total_memory = (size_t)8 * 1024 * 1024 * 1024;
    int idx = temper_device_register(TEMPER_DEVICE_GPU_0, gpu_caps);
    ASSERT(idx >= 0);

    TemperDeviceCaps fetched = temper_device_get_caps(TEMPER_DEVICE_GPU_0);
    ASSERT(fetched.supports_fp16 == true);
}

TEST(test_resource_lifecycle)
{
    TemperResource *res = temper_resource_create(TEMPER_DEVICE_CPU_0, 1024);
    ASSERT(res != NULL);
    ASSERT(res->bytes == 1024);
    ASSERT(res->refcount == 1);
    ASSERT(res->host_ptr != NULL);

    temper_resource_retain(res);
    ASSERT(res->refcount == 2);
    temper_resource_release(res);
    ASSERT(res->refcount == 1);
    temper_resource_release(res);
}

TEST(test_tensor_resource_indirection)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(t.resource != NULL);
    ASSERT(temper_device_equal(temper_tensor_device(&t), TEMPER_DEVICE_CPU_0));

    float *data = temper_tensor_data(&t);
    ASSERT(data != NULL);
    data[0] = 42.0f;
    ASSERT_FLOAT(temper_tensor_get(&t, 0), 42.0f, 1e-6f);

    temper_tensor_destroy(&t);
}

TEST(test_tensor_to_device_mobility)
{
    TemperShape s = temper_shape_1d(4);
    TemperTensor cpu_tensor = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *cdata = temper_tensor_data(&cpu_tensor);
    cdata[0] = 1.0f; cdata[1] = 2.0f; cdata[2] = 3.0f; cdata[3] = 4.0f;

    TemperDevice gpu0 = TEMPER_DEVICE_GPU_0;
    TemperTensor gpu_tensor = temper_tensor_to(&cpu_tensor, gpu0);

    ASSERT(temper_device_equal(temper_tensor_device(&gpu_tensor), gpu0));
    float *gdata = temper_tensor_data(&gpu_tensor);
    ASSERT_FLOAT(gdata[0], 1.0f, 1e-6f);
    ASSERT_FLOAT(gdata[3], 4.0f, 1e-6f);

    temper_tensor_destroy(&cpu_tensor);
    temper_tensor_destroy(&gpu_tensor);
}

int main(void)
{
    printf("=== Tensor Mobility Tests ===\n");
    RUN(test_device_table_and_caps);
    RUN(test_resource_lifecycle);
    RUN(test_tensor_resource_indirection);
    RUN(test_tensor_to_device_mobility);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
