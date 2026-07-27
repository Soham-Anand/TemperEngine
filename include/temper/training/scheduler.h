#ifndef TEMPER_SCHEDULER_H
#define TEMPER_SCHEDULER_H

#include "temper/training/optimizer.h"

typedef struct TemperScheduler
{
    float initial_lr;
    float min_lr;
    uint32_t warmup_steps;
    uint32_t total_steps;
} TemperScheduler;

TemperScheduler temper_scheduler_cosine(float initial_lr, float min_lr, uint32_t warmup_steps,
                                        uint32_t total_steps);
float temper_scheduler_get_lr(const TemperScheduler *sched, uint32_t step);
void temper_scheduler_step(TemperScheduler *sched, TemperOptimizer *opt, uint32_t step);

#endif
