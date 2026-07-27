#ifndef TEMPER_TRAINER_H
#define TEMPER_TRAINER_H

#include "temper/math/tensor.h"
#include "temper/training/optimizer.h"

typedef struct TemperMetrics
{
    float loss;
    float accuracy;
    uint64_t samples_seen;
} TemperMetrics;

typedef struct TemperTrainer
{
    TemperOptimizer *optimizer;
    TemperMetrics metrics;
    uint32_t epoch;
    uint32_t batch_size;
} TemperTrainer;

TemperTrainer temper_trainer_create(TemperOptimizer *opt, uint32_t batch_size);
void temper_trainer_zero_grad(TemperTrainer *trainer);
void temper_trainer_step(TemperTrainer *trainer);
void temper_trainer_epoch(TemperTrainer *trainer);
void temper_trainer_destroy(TemperTrainer *trainer);

#endif
