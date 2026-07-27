#include "temper/training/trainer.h"
#include "temper/core/logger.h"
#include <stdlib.h>

TemperTrainer temper_trainer_create(TemperOptimizer *opt, uint32_t batch_size)
{
    TemperTrainer trainer = {0};
    trainer.optimizer = opt;
    trainer.batch_size = batch_size;
    trainer.epoch = 0;
    trainer.metrics.loss = 0.0f;
    trainer.metrics.accuracy = 0.0f;
    trainer.metrics.samples_seen = 0;
    return trainer;
}

void temper_trainer_zero_grad(TemperTrainer *trainer)
{
    (void)trainer;
}

void temper_trainer_step(TemperTrainer *trainer)
{
    trainer->metrics.samples_seen += trainer->batch_size;
}

void temper_trainer_epoch(TemperTrainer *trainer)
{
    trainer->epoch++;
    temper_info("Epoch %u complete — loss: %.4f, samples: %lu", trainer->epoch,
                trainer->metrics.loss, (unsigned long)trainer->metrics.samples_seen);
}

void temper_trainer_destroy(TemperTrainer *trainer)
{
    (void)trainer;
}
