#ifndef TEMPER_GRAPH_H
#define TEMPER_GRAPH_H

#include <stdint.h>
#include "temper/math/tensor.h"

typedef enum TemperOpType
{
    TEMPER_OP_ADD,
    TEMPER_OP_SUB,
    TEMPER_OP_MUL,
    TEMPER_OP_DIV,
    TEMPER_OP_MATMUL,
    TEMPER_OP_RELU,
    TEMPER_OP_GELU,
    TEMPER_OP_SOFTMAX,
    TEMPER_OP_MEAN,
    TEMPER_OP_SUM
} TemperOpType;

typedef struct TemperGraphNode
{
    uint32_t id;
    TemperOpType op;
    TemperTensor *inputs[4];
    uint32_t input_count;
    TemperTensor output;
    struct TemperGraphNode **children;
    uint32_t child_count;
} TemperGraphNode;

typedef struct TemperTape
{
    TemperGraphNode **nodes;
    uint32_t node_count;
    uint32_t capacity;
} TemperTape;

TemperTape temper_tape_create(void);
void temper_tape_destroy(TemperTape *tape);
TemperGraphNode *temper_tape_record(TemperTape *tape, TemperOpType op, TemperTensor *inputs,
                                    uint32_t input_count);
void temper_tape_backward(TemperTape *tape);
void temper_tape_zero_grad(TemperTape *tape);

#endif
