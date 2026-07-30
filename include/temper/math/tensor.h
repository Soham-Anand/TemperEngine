#ifndef TEMPER_TENSOR_H
#define TEMPER_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "temper/core/memory.h"
#include "temper/math/shape.h"

typedef enum TemperDType
{
    TEMPER_DTYPE_F32,
    TEMPER_DTYPE_F16,
    TEMPER_DTYPE_BF16,
    TEMPER_DTYPE_I8,
    TEMPER_DTYPE_U8
} TemperDType;

typedef struct TemperTensor
{
    float *data;
    TemperShape shape;
    TemperDType dtype;
    int64_t strides[TEMPER_MAX_DIMS];
    bool owns_data;
} TemperTensor;

size_t temper_dtype_size(TemperDType dtype);

TemperTensor temper_tensor_create(TemperShape shape, TemperDType dtype);
TemperTensor temper_tensor_from_data(float *data, TemperShape shape, TemperDType dtype);
void temper_tensor_destroy(TemperTensor *t);

// Flat indexing (existing)
float temper_tensor_get(const TemperTensor *t, size_t idx);
void temper_tensor_set(TemperTensor *t, size_t idx, float val);

// Multi-dimensional indexing with strides
size_t temper_tensor_index(const TemperTensor *t, int64_t *indices);
float temper_tensor_get_nd(const TemperTensor *t, int64_t *indices);
void temper_tensor_set_nd(TemperTensor *t, int64_t *indices, float val);

// Shape queries
size_t temper_tensor_bytes(const TemperTensor *t);
bool temper_tensor_is_contiguous(const TemperTensor *t);

TemperTensor temper_tensor_add(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_sub(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_mul(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_div(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_matmul(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_transpose(const TemperTensor *t);
TemperTensor temper_tensor_reshape(const TemperTensor *t, TemperShape new_shape);
TemperTensor temper_tensor_sum(const TemperTensor *t, int axis);

#endif
