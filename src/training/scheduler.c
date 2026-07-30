#include "temper/training/scheduler.h"
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

TemperScheduler temper_scheduler_cosine(float initial_lr, float min_lr, uint32_t warmup_steps,
                                        uint32_t total_steps)
{
    TemperScheduler sched = {0};
    sched.initial_lr = initial_lr;
    sched.min_lr = min_lr;
    sched.warmup_steps = warmup_steps;
    sched.total_steps = total_steps;
    return sched;
}

float temper_scheduler_get_lr(const TemperScheduler *sched, uint32_t step)
{
    if (step < sched->warmup_steps)
    {
        return sched->initial_lr * ((float)step / (float)sched->warmup_steps);
    }
    if (step >= sched->total_steps)
    {
        return sched->min_lr;
    }
    float progress = (float)(step - sched->warmup_steps) /
                     (float)(sched->total_steps - sched->warmup_steps);
    return sched->min_lr + 0.5f * (sched->initial_lr - sched->min_lr) *
                               (1.0f + cosf((float)M_PI * progress));
}

void temper_scheduler_step(TemperScheduler *sched, TemperOptimizer *opt, uint32_t step)
{
    opt->learning_rate = temper_scheduler_get_lr(sched, step);
}
