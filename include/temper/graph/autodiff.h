#ifndef TEMPER_AUTODIFF_H
#define TEMPER_AUTODIFF_H

#include "temper/graph/tape.h"

void temper_autodiff_backward(TemperGraphNode *node);
void temper_autodiff_accumulate(TemperGraphNode *node, const TemperTensor *grad);
TemperTensor temper_autodiff_grad(const TemperGraphNode *node);

#endif
