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
    size_t stride;
    bool owns_data;
} TemperTensor;

size_t temper_dtype_size(TemperDType dtype);

TemperTensor temper_tensor_create(TemperShape shape, TemperDType dtype);
TemperTensor temper_tensor_from_data(float *data, TemperShape shape, TemperDType dtype);
void temper_tensor_destroy(TemperTensor *t);
float temper_tensor_get(const TemperTensor *t, size_t idx);
void temper_tensor_set(TemperTensor *t, size_t idx, float val);

TemperTensor temper_tensor_add(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_sub(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_mul(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_div(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_matmul(const TemperTensor *a, const TemperTensor *b);
TemperTensor temper_tensor_transpose(const TemperTensor *t);
TemperTensor temper_tensor_reshape(const TemperTensor *t, TemperShape new_shape);
TemperTensor temper_tensor_sum(const TemperTensor *t, int axis);

#endif
