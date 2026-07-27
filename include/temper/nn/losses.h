#ifndef TEMPER_LOSSES_H
#define TEMPER_LOSSES_H

#include "temper/math/tensor.h"

float temper_cross_entropy(const TemperTensor *predictions, const TemperTensor *targets);
float temper_mse(const TemperTensor *predictions, const TemperTensor *targets);

#endif
