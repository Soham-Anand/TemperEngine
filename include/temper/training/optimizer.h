#ifndef TEMPER_OPTIMIZER_H
#define TEMPER_OPTIMIZER_H

#include "temper/math/tensor.h"

typedef struct TemperOptimizer
{
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    uint32_t step;
    TemperTensor *m;
    TemperTensor *v;
} TemperOptimizer;

TemperOptimizer temper_optimizer_sgd(float lr);
TemperOptimizer temper_optimizer_adam(float lr, float beta1, float beta2, float epsilon);
void temper_optimizer_step(TemperOptimizer *opt, TemperTensor *params, const TemperTensor *grads);
void temper_optimizer_destroy(TemperOptimizer *opt);

#endif
