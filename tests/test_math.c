#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include <stdio.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) static void name(void)
#define RUN(name)                                                             \
    do                                                                        \
    {                                                                         \
        tests_run++;                                                          \
        printf("  %-40s", #name);                                             \
        name();                                                               \
        tests_passed++;                                                       \
        printf("PASS\n");                                                     \
    } while (0)

#define ASSERT(cond)                                                          \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            return;                                                           \
        }                                                                     \
    } while (0)

#define ASSERT_FLOAT(a, b, eps) (fabsf((a) - (b)) < (eps))

TEST(test_shape_1d)
{
    TemperShape s = temper_shape_1d(10);
    ASSERT(s.ndim == 1);
    ASSERT(s.dims[0] == 10);
}

TEST(test_shape_2d)
{
    TemperShape s = temper_shape_2d(3, 4);
    ASSERT(s.ndim == 2);
    ASSERT(s.dims[0] == 3);
    ASSERT(s.dims[1] == 4);
}

TEST(test_shape_count)
{
    TemperShape s = temper_shape_2d(3, 4);
    ASSERT(temper_shape_count(&s) == 12);
}

TEST(test_shape_equal)
{
    TemperShape a = temper_shape_2d(3, 4);
    TemperShape b = temper_shape_2d(3, 4);
    TemperShape c = temper_shape_2d(4, 3);
    ASSERT(temper_shape_equal(&a, &b));
    ASSERT(!temper_shape_equal(&a, &c));
}

TEST(test_tensor_create)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(t.data != NULL);
    ASSERT(t.shape.ndim == 2);
    ASSERT(t.owns_data);
    temper_tensor_destroy(&t);
}

TEST(test_tensor_add)
{
    TemperShape s = temper_shape_1d(3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(s, TEMPER_DTYPE_F32);
    a.data[0] = 1.0f;
    a.data[1] = 2.0f;
    a.data[2] = 3.0f;
    b.data[0] = 4.0f;
    b.data[1] = 5.0f;
    b.data[2] = 6.0f;
    TemperTensor c = temper_tensor_add(&a, &b);
    ASSERT_FLOAT(c.data[0], 5.0f, 1e-6f);
    ASSERT_FLOAT(c.data[1], 7.0f, 1e-6f);
    ASSERT_FLOAT(c.data[2], 9.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_mul)
{
    TemperShape s = temper_shape_1d(2);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(s, TEMPER_DTYPE_F32);
    a.data[0] = 2.0f;
    a.data[1] = 3.0f;
    b.data[0] = 4.0f;
    b.data[1] = 5.0f;
    TemperTensor c = temper_tensor_mul(&a, &b);
    ASSERT_FLOAT(c.data[0], 8.0f, 1e-6f);
    ASSERT_FLOAT(c.data[1], 15.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_matmul)
{
    TemperShape sa = temper_shape_2d(2, 3);
    TemperShape sb = temper_shape_2d(3, 2);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    float va[] = {1, 2, 3, 4, 5, 6};
    float vb[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++)
    {
        a.data[i] = va[i];
        b.data[i] = vb[i];
    }
    TemperTensor c = temper_tensor_matmul(&a, &b);
    ASSERT(c.shape.dims[0] == 2);
    ASSERT(c.shape.dims[1] == 2);
    ASSERT_FLOAT(c.data[0], 58.0f, 1e-5f);
    ASSERT_FLOAT(c.data[1], 64.0f, 1e-5f);
    ASSERT_FLOAT(c.data[2], 139.0f, 1e-5f);
    ASSERT_FLOAT(c.data[3], 154.0f, 1e-5f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_transpose)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    a.data[3] = 4;
    a.data[4] = 5;
    a.data[5] = 6;
    TemperTensor b = temper_tensor_transpose(&a);
    ASSERT(b.shape.dims[0] == 3);
    ASSERT(b.shape.dims[1] == 2);
    ASSERT_FLOAT(b.data[0], 1.0f, 1e-6f);
    ASSERT_FLOAT(b.data[1], 4.0f, 1e-6f);
    ASSERT_FLOAT(b.data[2], 2.0f, 1e-6f);
    ASSERT_FLOAT(b.data[3], 5.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_sum)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    a.data[3] = 4;
    a.data[4] = 5;
    a.data[5] = 6;
    TemperTensor b = temper_tensor_sum(&a, -1);
    ASSERT_FLOAT(b.data[0], 21.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

int main(void)
{
    printf("=== Math Tests ===\n");
    RUN(test_shape_1d);
    RUN(test_shape_2d);
    RUN(test_shape_count);
    RUN(test_shape_equal);
    RUN(test_tensor_create);
    RUN(test_tensor_add);
    RUN(test_tensor_mul);
    RUN(test_tensor_matmul);
    RUN(test_tensor_transpose);
    RUN(test_tensor_sum);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
