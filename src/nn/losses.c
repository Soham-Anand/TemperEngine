#include "temper/nn/losses.h"
#include "temper/utils/assert.h"
#include <math.h>

float temper_cross_entropy(const TemperTensor *predictions, const TemperTensor *targets)
{
    TEMPER_ASSERT(predictions->shape.ndim == 2);
    TEMPER_ASSERT(temper_shape_equal(&predictions->shape, &targets->shape));
    int64_t batch = predictions->shape.dims[0];
    int64_t classes = predictions->shape.dims[1];

    float *pred_data = temper_tensor_data(predictions);
    float *target_data = temper_tensor_data(targets);

    float loss = 0.0f;
    for (int64_t i = 0; i < batch; i++)
    {
        for (int64_t j = 0; j < classes; j++)
        {
            if (target_data[i * classes + j] > 0.0f)
            {
                float p = pred_data[i * classes + j];
                p = fmaxf(p, 1e-7f);
                loss -= target_data[i * classes + j] * logf(p);
            }
        }
    }
    return loss / (float)batch;
}

float temper_mse(const TemperTensor *predictions, const TemperTensor *targets)
{
    TEMPER_ASSERT(temper_shape_equal(&predictions->shape, &targets->shape));
    size_t count = temper_shape_count(&predictions->shape);
    float *pred_data = temper_tensor_data(predictions);
    float *target_data = temper_tensor_data(targets);

    float loss = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        float diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    return loss / (float)count;
}
