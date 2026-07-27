#include "temper/math/tensor.h"
#include "temper/utils/assert.h"
#include "temper/core/logger.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

size_t temper_dtype_size(TemperDType dtype)
{
    switch (dtype)
    {
    case TEMPER_DTYPE_F32:
        return 4;
    case TEMPER_DTYPE_F16:
        return 2;
    case TEMPER_DTYPE_BF16:
        return 2;
    case TEMPER_DTYPE_I8:
        return 1;
    case TEMPER_DTYPE_U8:
        return 1;
    default:
        return 0;
    }
}

TemperTensor temper_tensor_create(TemperShape shape, TemperDType dtype)
{
    TemperTensor t = {0};
    t.shape = shape;
    t.dtype = dtype;
    t.stride = (size_t)shape.dims[shape.ndim - 1];
    size_t count = temper_shape_count(&shape);
    t.data = (float *)calloc(count, sizeof(float));
    TEMPER_ASSERT_MSG(t.data != NULL, "Tensor allocation failed");
    t.owns_data = true;
    return t;
}

TemperTensor temper_tensor_from_data(float *data, TemperShape shape, TemperDType dtype)
{
    TemperTensor t = {0};
    t.shape = shape;
    t.dtype = dtype;
    t.stride = (size_t)shape.dims[shape.ndim - 1];
    t.data = data;
    t.owns_data = false;
    return t;
}

void temper_tensor_destroy(TemperTensor *t)
{
    if (t->owns_data && t->data)
    {
        free(t->data);
        t->data = NULL;
    }
}

float temper_tensor_get(const TemperTensor *t, size_t idx)
{
    return t->data[idx];
}

void temper_tensor_set(TemperTensor *t, size_t idx, float val)
{
    t->data[idx] = val;
}

TemperTensor temper_tensor_add(const TemperTensor *a, const TemperTensor *b)
{
    TEMPER_ASSERT(temper_shape_equal(&a->shape, &b->shape));
    TemperTensor result = temper_tensor_create(a->shape, a->dtype);
    size_t count = temper_shape_count(&a->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = a->data[i] + b->data[i];
    }
    return result;
}

TemperTensor temper_tensor_sub(const TemperTensor *a, const TemperTensor *b)
{
    TEMPER_ASSERT(temper_shape_equal(&a->shape, &b->shape));
    TemperTensor result = temper_tensor_create(a->shape, a->dtype);
    size_t count = temper_shape_count(&a->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = a->data[i] - b->data[i];
    }
    return result;
}

TemperTensor temper_tensor_mul(const TemperTensor *a, const TemperTensor *b)
{
    TEMPER_ASSERT(temper_shape_equal(&a->shape, &b->shape));
    TemperTensor result = temper_tensor_create(a->shape, a->dtype);
    size_t count = temper_shape_count(&a->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = a->data[i] * b->data[i];
    }
    return result;
}

TemperTensor temper_tensor_div(const TemperTensor *a, const TemperTensor *b)
{
    TEMPER_ASSERT(temper_shape_equal(&a->shape, &b->shape));
    TemperTensor result = temper_tensor_create(a->shape, a->dtype);
    size_t count = temper_shape_count(&a->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = a->data[i] / b->data[i];
    }
    return result;
}

TemperTensor temper_tensor_matmul(const TemperTensor *a, const TemperTensor *b)
{
    TEMPER_ASSERT(a->shape.ndim == 2 && b->shape.ndim == 2);
    TEMPER_ASSERT(a->shape.dims[1] == b->shape.dims[0]);
    int64_t m = a->shape.dims[0];
    int64_t k = a->shape.dims[1];
    int64_t n = b->shape.dims[1];
    TemperShape out_shape = temper_shape_2d(m, n);
    TemperTensor result = temper_tensor_create(out_shape, a->dtype);
    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++)
            {
                sum += a->data[i * k + l] * b->data[l * n + j];
            }
            result.data[i * n + j] = sum;
        }
    }
    return result;
}

TemperTensor temper_tensor_transpose(const TemperTensor *t)
{
    TEMPER_ASSERT(t->shape.ndim == 2);
    int64_t rows = t->shape.dims[0];
    int64_t cols = t->shape.dims[1];
    TemperShape out_shape = temper_shape_2d(cols, rows);
    TemperTensor result = temper_tensor_create(out_shape, t->dtype);
    for (int64_t i = 0; i < rows; i++)
    {
        for (int64_t j = 0; j < cols; j++)
        {
            result.data[j * rows + i] = t->data[i * cols + j];
        }
    }
    return result;
}

TemperTensor temper_tensor_reshape(const TemperTensor *t, TemperShape new_shape)
{
    TEMPER_ASSERT(temper_shape_count(&t->shape) == temper_shape_count(&new_shape));
    TemperTensor result = temper_tensor_from_data(t->data, new_shape, t->dtype);
    result.owns_data = false;
    return result;
}

TemperTensor temper_tensor_sum(const TemperTensor *t, int axis)
{
    if (axis < 0)
    {
        TemperShape s = temper_shape_1d(1);
        TemperTensor result = temper_tensor_create(s, t->dtype);
        size_t count = temper_shape_count(&t->shape);
        float sum = 0.0f;
        for (size_t i = 0; i < count; i++)
        {
            sum += t->data[i];
        }
        result.data[0] = sum;
        return result;
    }
    TEMPER_ASSERT(axis < t->shape.ndim);
    TemperShape out_shape = {0};
    out_shape.ndim = t->shape.ndim - 1;
    int oi = 0;
    for (int i = 0; i < t->shape.ndim; i++)
    {
        if (i != axis)
        {
            out_shape.dims[oi++] = t->shape.dims[i];
        }
    }
    TemperTensor result = temper_tensor_create(out_shape, t->dtype);
    size_t out_count = temper_shape_count(&out_shape);
    size_t axis_size = (size_t)t->shape.dims[axis];
    for (size_t i = 0; i < out_count; i++)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < axis_size; j++)
        {
            sum += t->data[i * axis_size + j];
        }
        result.data[i] = sum;
    }
    return result;
}
