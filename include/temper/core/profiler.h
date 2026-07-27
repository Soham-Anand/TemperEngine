#ifndef TEMPER_PROFILER_H
#define TEMPER_PROFILER_H

#include <stddef.h>
#include <stdint.h>

typedef struct TemperProfiler
{
    uint64_t alloc_total;
    uint64_t alloc_count;
    uint64_t peak_usage;
    uint64_t current_usage;
} TemperProfiler;

void temper_profiler_init(TemperProfiler *prof);
void temper_profiler_reset(TemperProfiler *prof);
void temper_profiler_record_alloc(TemperProfiler *prof, size_t size);
void temper_profiler_record_free(TemperProfiler *prof, size_t size);
void temper_profiler_print(const TemperProfiler *prof);

#endif
