#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include <stdio.h>
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
    ASSERT(temper_tensor_data(&t) != NULL);
    ASSERT(t.shape.ndim == 2);
    temper_tensor_destroy(&t);
}

TEST(test_tensor_add)
{
    TemperShape s = temper_shape_1d(3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 1.0f; adata[1] = 2.0f; adata[2] = 3.0f;
    bdata[0] = 4.0f; bdata[1] = 5.0f; bdata[2] = 6.0f;
    TemperTensor c = temper_tensor_add(&a, &b);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 5.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 7.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 2), 9.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_mul)
{
    TemperShape s = temper_shape_1d(2);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 2.0f; adata[1] = 3.0f;
    bdata[0] = 4.0f; bdata[1] = 5.0f;
    TemperTensor c = temper_tensor_mul(&a, &b);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 8.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 15.0f, 1e-6f);
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
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    float va[] = {1, 2, 3, 4, 5, 6};
    float vb[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++)
    {
        adata[i] = va[i];
        bdata[i] = vb[i];
    }
    TemperTensor c = temper_tensor_matmul(&a, &b);
    ASSERT(c.shape.dims[0] == 2);
    ASSERT(c.shape.dims[1] == 2);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 58.0f, 1e-5f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 64.0f, 1e-5f);
    ASSERT_FLOAT(temper_tensor_get(&c, 2), 139.0f, 1e-5f);
    ASSERT_FLOAT(temper_tensor_get(&c, 3), 154.0f, 1e-5f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_transpose)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    TemperTensor b = temper_tensor_transpose(&a);
    ASSERT(b.shape.dims[0] == 3);
    ASSERT(b.shape.dims[1] == 2);
    ASSERT_FLOAT(temper_tensor_get(&b, 0), 1.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 1), 4.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 2), 2.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 3), 5.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_sum)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    TemperTensor b = temper_tensor_sum(&a, -1);
    ASSERT_FLOAT(temper_tensor_get(&b, 0), 21.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_sum_axis0)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    TemperTensor b = temper_tensor_sum(&a, 0);
    ASSERT(b.shape.ndim == 1);
    ASSERT(b.shape.dims[0] == 3);
    ASSERT_FLOAT(temper_tensor_get(&b, 0), 5.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 1), 7.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 2), 9.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_sum_axis1)
{
    TemperShape s = temper_shape_2d(2, 3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    TemperTensor b = temper_tensor_sum(&a, 1);
    ASSERT(b.shape.ndim == 1);
    ASSERT(b.shape.dims[0] == 2);
    ASSERT_FLOAT(temper_tensor_get(&b, 0), 6.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 1), 15.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_broadcast_2d_1d)
{
    TemperShape sa = temper_shape_2d(2, 3);
    TemperShape sb = temper_shape_1d(3);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    bdata[0] = 10; bdata[1] = 20; bdata[2] = 30;
    TemperTensor c = temper_tensor_add(&a, &b);
    ASSERT(c.shape.ndim == 2);
    ASSERT(c.shape.dims[0] == 2);
    ASSERT(c.shape.dims[1] == 3);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 11.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 22.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 2), 33.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 3), 14.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 4), 25.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 5), 36.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_broadcast_2d_col)
{
    TemperShape sa = temper_shape_2d(2, 3);
    TemperShape sb = temper_shape_2d(2, 1);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    bdata[0] = 10; bdata[1] = 20;
    TemperTensor c = temper_tensor_mul(&a, &b);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 10.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 20.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 2), 30.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 3), 80.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 4), 100.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 5), 120.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_broadcast_scalar)
{
    TemperShape sa = temper_shape_2d(2, 3);
    TemperShape sb = temper_shape_2d(1, 1);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(sb, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 1; adata[1] = 2; adata[2] = 3;
    adata[3] = 4; adata[4] = 5; adata[5] = 6;
    bdata[0] = 100;
    TemperTensor c = temper_tensor_sub(&a, &b);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), -99.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 5), -94.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_nd_index)
{
    TemperShape s = temper_shape_2d(3, 4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    int64_t idx[] = {2, 3};
    temper_tensor_set_nd(&t, idx, 42.0f);
    float val = temper_tensor_get_nd(&t, idx);
    ASSERT_FLOAT(val, 42.0f, 1e-6f);
    temper_tensor_destroy(&t);
}

TEST(test_tensor_bytes)
{
    TemperShape s = temper_shape_2d(3, 4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(temper_tensor_bytes(&t) == 12 * sizeof(float));
    temper_tensor_destroy(&t);
}

TEST(test_tensor_is_contiguous)
{
    TemperShape s = temper_shape_2d(3, 4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(temper_tensor_is_contiguous(&t));
    TemperTensor tr = temper_tensor_transpose(&t);
    ASSERT(temper_tensor_is_contiguous(&tr));
    temper_tensor_destroy(&t);
    temper_tensor_destroy(&tr);
}

TEST(test_tensor_div)
{
    TemperShape s = temper_shape_1d(3);
    TemperTensor a = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor b = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    float *bdata = temper_tensor_data(&b);
    adata[0] = 10.0f; adata[1] = 6.0f; adata[2] = 4.0f;
    bdata[0] = 2.0f;  bdata[1] = 3.0f; bdata[2] = 5.0f;
    TemperTensor c = temper_tensor_div(&a, &b);
    ASSERT_FLOAT(temper_tensor_get(&c, 0), 5.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 1), 2.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&c, 2), 0.8f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
    temper_tensor_destroy(&c);
}

TEST(test_tensor_reshape)
{
    TemperShape sa = temper_shape_2d(2, 6);
    TemperTensor a = temper_tensor_create(sa, TEMPER_DTYPE_F32);
    float *adata = temper_tensor_data(&a);
    for (int i = 0; i < 12; i++)
    {
        adata[i] = (float)i;
    }
    TemperShape sb = temper_shape_3d(2, 3, 2);
    TemperTensor b = temper_tensor_reshape(&a, sb);
    ASSERT(b.shape.ndim == 3);
    ASSERT(b.shape.dims[0] == 2);
    ASSERT(b.shape.dims[1] == 3);
    ASSERT(b.shape.dims[2] == 2);
    ASSERT_FLOAT(temper_tensor_get(&b, 0), 0.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 5), 5.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&b, 11), 11.0f, 1e-6f);
    temper_tensor_destroy(&a);
    temper_tensor_destroy(&b);
}

TEST(test_tensor_from_data)
{
    TemperShape s = temper_shape_2d(2, 2);
    float raw[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TemperTensor t = temper_tensor_from_data(raw, s, TEMPER_DTYPE_F32);
    // Phase 2: from_data copies into a managed resource
    ASSERT(temper_tensor_data(&t) != NULL);
    ASSERT(temper_tensor_get(&t, 0) == 1.0f);
    ASSERT(temper_tensor_get(&t, 3) == 4.0f);
    // Destroying the tensor must not free the caller's buffer
    temper_tensor_destroy(&t);
    ASSERT(raw[0] == 1.0f);
    ASSERT(raw[3] == 4.0f);
}

TEST(test_dtype_size)
{
    ASSERT(temper_dtype_size(TEMPER_DTYPE_F32) == 4);
    ASSERT(temper_dtype_size(TEMPER_DTYPE_F16) == 2);
    ASSERT(temper_dtype_size(TEMPER_DTYPE_BF16) == 2);
    ASSERT(temper_dtype_size(TEMPER_DTYPE_I8) == 1);
    ASSERT(temper_dtype_size(TEMPER_DTYPE_U8) == 1);
}

TEST(test_tensor_get_set_flat)
{
    TemperShape s = temper_shape_1d(4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    temper_tensor_set(&t, 0, 10.0f);
    temper_tensor_set(&t, 1, 20.0f);
    temper_tensor_set(&t, 3, 40.0f);
    ASSERT_FLOAT(temper_tensor_get(&t, 0), 10.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&t, 1), 20.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&t, 2), 0.0f, 1e-6f);
    ASSERT_FLOAT(temper_tensor_get(&t, 3), 40.0f, 1e-6f);
    temper_tensor_destroy(&t);
}

TEST(test_tensor_zero_size)
{
    TemperShape s = temper_shape_2d(0, 5);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(temper_shape_count(&s) == 0);
    ASSERT(temper_tensor_bytes(&t) == 0);
    temper_tensor_destroy(&t);
}

TEST(test_tensor_index_nd)
{
    TemperShape s = temper_shape_3d(2, 3, 4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    int64_t idx[] = {1, 2, 3};
    size_t flat = temper_tensor_index(&t, idx);
    // 1*12 + 2*4 + 3 = 23
    ASSERT(flat == 23);
    temper_tensor_destroy(&t);
}

TEST(test_shape_3d)
{
    TemperShape s = temper_shape_3d(2, 3, 4);
    ASSERT(s.ndim == 3);
    ASSERT(s.dims[0] == 2);
    ASSERT(s.dims[1] == 3);
    ASSERT(s.dims[2] == 4);
    ASSERT(temper_shape_count(&s) == 24);
}

TEST(test_shape_make)
{
    // Varargs must be promoted to int64_t (Windows LLP64: int is 32-bit)
    TemperShape s = temper_shape_make(3, (int64_t)2, (int64_t)4, (int64_t)8);
    ASSERT(s.ndim == 3);
    ASSERT(s.dims[0] == 2);
    ASSERT(s.dims[1] == 4);
    ASSERT(s.dims[2] == 8);
    ASSERT(temper_shape_count(&s) == 64);
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
    RUN(test_tensor_sum_axis0);
    RUN(test_tensor_sum_axis1);
    RUN(test_tensor_broadcast_2d_1d);
    RUN(test_tensor_broadcast_2d_col);
    RUN(test_tensor_broadcast_scalar);
    RUN(test_tensor_nd_index);
    RUN(test_tensor_bytes);
    RUN(test_tensor_is_contiguous);
    RUN(test_tensor_div);
    RUN(test_tensor_reshape);
    RUN(test_tensor_from_data);
    RUN(test_dtype_size);
    RUN(test_tensor_get_set_flat);
    RUN(test_tensor_zero_size);
    RUN(test_tensor_index_nd);
    RUN(test_shape_3d);
    RUN(test_shape_make);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
