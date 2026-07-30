#include "temper/math/tensor.h"
#include "temper/utils/assert.h"
#include "temper/core/logger.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Check if two shapes are broadcast-compatible
static TemperShape broadcast_shape(const TemperShape *a, const TemperShape *b)
{
    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    TemperShape result = {0};
    result.ndim = (uint8_t)max_ndim;
    for (int i = 0; i < max_ndim; i++)
    {
        int64_t da = (i < max_ndim - a->ndim) ? 1 : a->dims[i - (max_ndim - a->ndim)];
        int64_t db = (i < max_ndim - b->ndim) ? 1 : b->dims[i - (max_ndim - b->ndim)];
        result.dims[i] = da > db ? da : db;
    }
    return result;
}

// Get element from tensor with broadcast indexing
static float broadcast_get(const TemperTensor *t, const TemperShape *out_shape, size_t flat_idx)
{
    // Convert flat index to multi-dimensional index in output shape
    int64_t idx[TEMPER_MAX_DIMS];
    size_t remaining = flat_idx;
    for (int i = out_shape->ndim - 1; i >= 0; i--)
    {
        idx[i] = remaining % out_shape->dims[i];
        remaining /= out_shape->dims[i];
    }

    // Map back to tensor's shape (broadcasting: dimension 1 repeats)
    size_t tensor_idx = 0;
    int offset = out_shape->ndim - t->shape.ndim;
    for (int i = 0; i < t->shape.ndim; i++)
    {
        int64_t dim_idx = (i + offset >= 0) ? idx[i + offset] : 0;
        // If tensor dimension is 1, always use index 0 (broadcast)
        if (t->shape.dims[i] == 1)
        {
            dim_idx = 0;
        }
        tensor_idx = tensor_idx * (size_t)t->shape.dims[i] + (size_t)dim_idx;
    }

    return t->data[tensor_idx];
}

static void compute_strides(const TemperShape *shape, int64_t *strides)
{
    if (shape->ndim == 0)
    {
        return;
    }
    strides[shape->ndim - 1] = 1;
    for (int i = shape->ndim - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape->dims[i + 1];
    }
}

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
    compute_strides(&shape, t.strides);
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
    compute_strides(&shape, t.strides);
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

size_t temper_tensor_index(const TemperTensor *t, int64_t *indices)
{
    size_t idx = 0;
    for (int i = 0; i < t->shape.ndim; i++)
    {
        idx += (size_t)(indices[i] * t->strides[i]);
    }
    return idx;
}

float temper_tensor_get_nd(const TemperTensor *t, int64_t *indices)
{
    return t->data[temper_tensor_index(t, indices)];
}

void temper_tensor_set_nd(TemperTensor *t, int64_t *indices, float val)
{
    t->data[temper_tensor_index(t, indices)] = val;
}

size_t temper_tensor_bytes(const TemperTensor *t)
{
    return temper_shape_count(&t->shape) * temper_dtype_size(t->dtype);
}

bool temper_tensor_is_contiguous(const TemperTensor *t)
{
    if (t->shape.ndim == 0) return true;
    int64_t expected_stride = 1;
    for (int i = t->shape.ndim - 1; i >= 0; i--)
    {
        if (t->strides[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= t->shape.dims[i];
    }
    return true;
}

TemperTensor temper_tensor_add(const TemperTensor *a, const TemperTensor *b)
{
    TemperShape out_shape = broadcast_shape(&a->shape, &b->shape);
    TemperTensor result = temper_tensor_create(out_shape, a->dtype);
    size_t count = temper_shape_count(&out_shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = broadcast_get(a, &out_shape, i) + broadcast_get(b, &out_shape, i);
    }
    return result;
}

TemperTensor temper_tensor_sub(const TemperTensor *a, const TemperTensor *b)
{
    TemperShape out_shape = broadcast_shape(&a->shape, &b->shape);
    TemperTensor result = temper_tensor_create(out_shape, a->dtype);
    size_t count = temper_shape_count(&out_shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = broadcast_get(a, &out_shape, i) - broadcast_get(b, &out_shape, i);
    }
    return result;
}

TemperTensor temper_tensor_mul(const TemperTensor *a, const TemperTensor *b)
{
    TemperShape out_shape = broadcast_shape(&a->shape, &b->shape);
    TemperTensor result = temper_tensor_create(out_shape, a->dtype);
    size_t count = temper_shape_count(&out_shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = broadcast_get(a, &out_shape, i) * broadcast_get(b, &out_shape, i);
    }
    return result;
}

TemperTensor temper_tensor_div(const TemperTensor *a, const TemperTensor *b)
{
    TemperShape out_shape = broadcast_shape(&a->shape, &b->shape);
    TemperTensor result = temper_tensor_create(out_shape, a->dtype);
    size_t count = temper_shape_count(&out_shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = broadcast_get(a, &out_shape, i) / broadcast_get(b, &out_shape, i);
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
    if (t->shape.ndim == 1)
    {
        out_shape = temper_shape_1d(1);
    }
    else
    {
        out_shape.ndim = t->shape.ndim - 1;
        int oi = 0;
        for (int i = 0; i < t->shape.ndim; i++)
        {
            if (i != axis)
            {
                out_shape.dims[oi++] = t->shape.dims[i];
            }
        }
    }

    TemperTensor result = temper_tensor_create(out_shape, t->dtype);
    size_t out_count = temper_shape_count(&out_shape);
    int64_t axis_size = t->shape.dims[axis];

    int64_t out_indices[TEMPER_MAX_DIMS];
    int64_t full_indices[TEMPER_MAX_DIMS];

    for (size_t i = 0; i < out_count; i++)
    {
        // Unravel flat out_idx 'i' to out_indices
        size_t remaining = i;
        for (int d = out_shape.ndim - 1; d >= 0; d--)
        {
            out_indices[d] = remaining % out_shape.dims[d];
            remaining /= out_shape.dims[d];
        }

        float sum = 0.0f;
        for (int64_t k = 0; k < axis_size; k++)
        {
            int oi_idx = 0;
            for (int d = 0; d < t->shape.ndim; d++)
            {
                if (d == axis)
                {
                    full_indices[d] = k;
                }
                else
                {
                    full_indices[d] = (out_shape.ndim == 1 && t->shape.ndim == 1) ? 0 : out_indices[oi_idx++];
                }
            }
            sum += temper_tensor_get_nd(t, full_indices);
        }
        result.data[i] = sum;
    }
    return result;
}
