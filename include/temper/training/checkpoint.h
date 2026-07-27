#ifndef TEMPER_CHECKPOINT_H
#define TEMPER_CHECKPOINT_H

#include <stdint.h>
#include <stdbool.h>

typedef struct TemperCheckpoint
{
    uint32_t epoch;
    uint32_t step;
    float loss;
} TemperCheckpoint;

int temper_checkpoint_save(const char *path, const TemperCheckpoint *cp);
int temper_checkpoint_load(const char *path, TemperCheckpoint *cp);
bool temper_checkpoint_exists(const char *path);

#endif
