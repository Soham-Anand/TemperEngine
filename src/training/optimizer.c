#include "temper/training/optimizer.h"
#include "temper/utils/assert.h"
#include <stdlib.h>
#include <math.h>

TemperOptimizer temper_optimizer_sgd(float lr)
{
    TemperOptimizer opt = {0};
    opt.learning_rate = lr;
    opt.beta1 = 0.0f;
    opt.beta2 = 0.0f;
    opt.epsilon = 0.0f;
    opt.step = 0;
    opt.m = NULL;
    opt.v = NULL;
    return opt;
}

TemperOptimizer temper_optimizer_adam(float lr, float beta1, float beta2, float epsilon)
{
    TemperOptimizer opt = {0};
    opt.learning_rate = lr;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.epsilon = epsilon;
    opt.step = 0;
    opt.m = NULL;
    opt.v = NULL;
    return opt;
}

void temper_optimizer_step(TemperOptimizer *opt, TemperTensor *params, const TemperTensor *grads)
{
    TEMPER_ASSERT(temper_shape_equal(&params->shape, &grads->shape));
    size_t count = temper_shape_count(&params->shape);
    opt->step++;

    float *p_data = temper_tensor_data(params);
    float *g_data = temper_tensor_data(grads);

    if (opt->beta1 > 0.0f)
    {
        if (!opt->m)
        {
            opt->m = (TemperTensor *)malloc(sizeof(TemperTensor));
            *opt->m = temper_tensor_create(params->shape, params->dtype);
            opt->v = (TemperTensor *)malloc(sizeof(TemperTensor));
            *opt->v = temper_tensor_create(params->shape, params->dtype);
        }
        float *m_data = temper_tensor_data(opt->m);
        float *v_data = temper_tensor_data(opt->v);
        float bc1 = 1.0f - powf(opt->beta1, (float)opt->step);
        float bc2 = 1.0f - powf(opt->beta2, (float)opt->step);
        for (size_t i = 0; i < count; i++)
        {
            m_data[i] = opt->beta1 * m_data[i] + (1.0f - opt->beta1) * g_data[i];
            v_data[i] = opt->beta2 * v_data[i] + (1.0f - opt->beta2) * g_data[i] * g_data[i];
            float m_hat = m_data[i] / bc1;
            float v_hat = v_data[i] / bc2;
            p_data[i] -= opt->learning_rate * m_hat / (sqrtf(v_hat) + opt->epsilon);
        }
    }
    else
    {
        for (size_t i = 0; i < count; i++)
        {
            p_data[i] -= opt->learning_rate * g_data[i];
        }
    }
}

void temper_optimizer_destroy(TemperOptimizer *opt)
{
    if (opt->m)
    {
        temper_tensor_destroy(opt->m);
        free(opt->m);
        opt->m = NULL;
    }
    if (opt->v)
    {
        temper_tensor_destroy(opt->v);
        free(opt->v);
        opt->v = NULL;
    }
}
