#include "temper/graph/autodiff.h"
#include "temper/core/logger.h"

void temper_autodiff_backward(TemperGraphNode *node)
{
    if (!node)
    {
        return;
    }
    temper_debug("Autodiff backward: node %u op %d", node->id, node->op);
}

void temper_autodiff_accumulate(TemperGraphNode *node, const TemperTensor *grad)
{
    if (!node || !grad)
    {
        return;
    }
    (void)grad;
    temper_debug("Autodiff accumulate: node %u", node->id);
}

TemperTensor temper_autodiff_grad(const TemperGraphNode *node)
{
    TemperTensor empty = {0};
    if (!node)
    {
        return empty;
    }
    return empty;
}
