#ifndef TEMPER_ACTIVATIONS_H
#define TEMPER_ACTIVATIONS_H

#include "temper/math/tensor.h"

TemperTensor temper_relu(const TemperTensor *input);
TemperTensor temper_gelu(const TemperTensor *input);
TemperTensor temper_silu(const TemperTensor *input);
TemperTensor temper_softmax(const TemperTensor *input, int axis);

#endif
