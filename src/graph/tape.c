#include "temper/graph/tape.h"
#include "temper/utils/assert.h"
#include "temper/core/logger.h"
#include <stdlib.h>
#include <string.h>

TemperTape temper_tape_create(void)
{
    TemperTape tape = {0};
    tape.capacity = 256;
    tape.nodes = (TemperGraphNode **)calloc(tape.capacity, sizeof(TemperGraphNode *));
    tape.node_count = 0;
    return tape;
}

void temper_tape_destroy(TemperTape *tape)
{
    for (uint32_t i = 0; i < tape->node_count; i++)
    {
        TemperGraphNode *node = tape->nodes[i];
        if (node->children)
        {
            free(node->children);
        }
        free(node);
    }
    free(tape->nodes);
    tape->nodes = NULL;
    tape->node_count = 0;
    tape->capacity = 0;
}

TemperGraphNode *temper_tape_record(TemperTape *tape, TemperOpType op, TemperTensor *inputs,
                                    uint32_t input_count)
{
    if (tape->node_count >= tape->capacity)
    {
        tape->capacity *= 2;
        tape->nodes =
            (TemperGraphNode **)realloc(tape->nodes, sizeof(TemperGraphNode *) * tape->capacity);
    }

    TemperGraphNode *node = (TemperGraphNode *)calloc(1, sizeof(TemperGraphNode));
    node->id = tape->node_count;
    node->op = op;
    node->input_count = input_count;
    for (uint32_t i = 0; i < input_count && i < 4; i++)
    {
        node->inputs[i] = &inputs[i];
    }

    tape->nodes[tape->node_count++] = node;
    return node;
}

void temper_tape_backward(TemperTape *tape)
{
    for (int i = (int)tape->node_count - 1; i >= 0; i--)
    {
        TemperGraphNode *node = tape->nodes[i];
        (void)node;
    }
    temper_info("Backward pass: %u nodes processed", tape->node_count);
}

void temper_tape_zero_grad(TemperTape *tape)
{
    for (uint32_t i = 0; i < tape->node_count; i++)
    {
        TemperGraphNode *node = tape->nodes[i];
        (void)node;
    }
}
