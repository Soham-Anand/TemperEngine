#include "temper/math/shape.h"
#include <stdarg.h>
#include <string.h>

TemperShape temper_shape_make(int ndim, ...)
{
    TemperShape s = {0};
    s.ndim = (uint8_t)ndim;
    va_list args;
    va_start(args, ndim);
    for (int i = 0; i < ndim && i < TEMPER_MAX_DIMS; i++)
    {
        s.dims[i] = va_arg(args, int64_t);
    }
    va_end(args);
    return s;
}

TemperShape temper_shape_1d(int64_t d0)
{
    TemperShape s = {0};
    s.ndim = 1;
    s.dims[0] = d0;
    return s;
}

TemperShape temper_shape_2d(int64_t d0, int64_t d1)
{
    TemperShape s = {0};
    s.ndim = 2;
    s.dims[0] = d0;
    s.dims[1] = d1;
    return s;
}

TemperShape temper_shape_3d(int64_t d0, int64_t d1, int64_t d2)
{
    TemperShape s = {0};
    s.ndim = 3;
    s.dims[0] = d0;
    s.dims[1] = d1;
    s.dims[2] = d2;
    return s;
}

size_t temper_shape_count(const TemperShape *s)
{
    size_t count = 1;
    for (uint8_t i = 0; i < s->ndim; i++)
    {
        count *= (size_t)s->dims[i];
    }
    return count;
}

bool temper_shape_equal(const TemperShape *a, const TemperShape *b)
{
    if (a->ndim != b->ndim)
    {
        return false;
    }
    for (uint8_t i = 0; i < a->ndim; i++)
    {
        if (a->dims[i] != b->dims[i])
        {
            return false;
        }
    }
    return true;
}
