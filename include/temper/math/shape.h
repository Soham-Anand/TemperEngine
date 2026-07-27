#ifndef TEMPER_SHAPE_H
#define TEMPER_SHAPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TEMPER_MAX_DIMS 8

typedef struct TemperShape
{
    int64_t dims[TEMPER_MAX_DIMS];
    uint8_t ndim;
} TemperShape;

TemperShape temper_shape_make(int ndim, ...);
TemperShape temper_shape_1d(int64_t d0);
TemperShape temper_shape_2d(int64_t d0, int64_t d1);
TemperShape temper_shape_3d(int64_t d0, int64_t d1, int64_t d2);
size_t temper_shape_count(const TemperShape *s);
bool temper_shape_equal(const TemperShape *a, const TemperShape *b);

#endif
