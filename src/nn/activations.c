#include "temper/nn/activations.h"
#include "temper/utils/assert.h"
#include <math.h>

TemperTensor temper_relu(const TemperTensor *input)
{
    TemperTensor result = temper_tensor_create(input->shape, input->dtype);
    size_t count = temper_shape_count(&input->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }
    return result;
}

static float gelu_approx(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

TemperTensor temper_gelu(const TemperTensor *input)
{
    TemperTensor result = temper_tensor_create(input->shape, input->dtype);
    size_t count = temper_shape_count(&input->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = gelu_approx(input->data[i]);
    }
    return result;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

TemperTensor temper_silu(const TemperTensor *input)
{
    TemperTensor result = temper_tensor_create(input->shape, input->dtype);
    size_t count = temper_shape_count(&input->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = input->data[i] * sigmoid(input->data[i]);
    }
    return result;
}

TemperTensor temper_softmax(const TemperTensor *input, int axis)
{
    TEMPER_ASSERT(input->shape.ndim == 2);
    TemperTensor result = temper_tensor_create(input->shape, input->dtype);
    int64_t rows = input->shape.dims[0];
    int64_t cols = input->shape.dims[1];
    if (axis == 1 || axis == -1)
    {
        for (int64_t i = 0; i < rows; i++)
        {
            float max_val = input->data[i * cols];
            for (int64_t j = 1; j < cols; j++)
            {
                if (input->data[i * cols + j] > max_val)
                {
                    max_val = input->data[i * cols + j];
                }
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < cols; j++)
            {
                result.data[i * cols + j] = expf(input->data[i * cols + j] - max_val);
                sum += result.data[i * cols + j];
            }
            for (int64_t j = 0; j < cols; j++)
            {
                result.data[i * cols + j] /= sum;
            }
        }
    }
    return result;
}
