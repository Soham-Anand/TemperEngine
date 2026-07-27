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

    if (opt->beta1 > 0.0f)
    {
        if (!opt->m)
        {
            opt->m = (TemperTensor *)malloc(sizeof(TemperTensor));
            *opt->m = temper_tensor_create(params->shape, params->dtype);
            opt->v = (TemperTensor *)malloc(sizeof(TemperTensor));
            *opt->v = temper_tensor_create(params->shape, params->dtype);
        }
        float bc1 = 1.0f - powf(opt->beta1, (float)opt->step);
        float bc2 = 1.0f - powf(opt->beta2, (float)opt->step);
        for (size_t i = 0; i < count; i++)
        {
            opt->m->data[i] = opt->beta1 * opt->m->data[i] + (1.0f - opt->beta1) * grads->data[i];
            opt->v->data[i] =
                opt->beta2 * opt->v->data[i] + (1.0f - opt->beta2) * grads->data[i] * grads->data[i];
            float m_hat = opt->m->data[i] / bc1;
            float v_hat = opt->v->data[i] / bc2;
            params->data[i] -= opt->learning_rate * m_hat / (sqrtf(v_hat) + opt->epsilon);
        }
    }
    else
    {
        for (size_t i = 0; i < count; i++)
        {
            params->data[i] -= opt->learning_rate * grads->data[i];
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
