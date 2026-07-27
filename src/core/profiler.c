#include "temper/core/profiler.h"
#include "temper/core/logger.h"
#include <stdio.h>

void temper_profiler_init(TemperProfiler *prof)
{
    prof->alloc_total = 0;
    prof->alloc_count = 0;
    prof->peak_usage = 0;
    prof->current_usage = 0;
}

void temper_profiler_reset(TemperProfiler *prof)
{
    prof->alloc_total = 0;
    prof->alloc_count = 0;
    prof->peak_usage = 0;
    prof->current_usage = 0;
}

void temper_profiler_record_alloc(TemperProfiler *prof, size_t size)
{
    prof->alloc_total += size;
    prof->alloc_count++;
    prof->current_usage += size;
    if (prof->current_usage > prof->peak_usage)
    {
        prof->peak_usage = prof->current_usage;
    }
}

void temper_profiler_record_free(TemperProfiler *prof, size_t size)
{
    if (prof->current_usage >= size)
    {
        prof->current_usage -= size;
    }
}

void temper_profiler_print(const TemperProfiler *prof)
{
    temper_info("Profiler: allocs=%lu total=%lu peak=%lu current=%lu", (unsigned long)prof->alloc_count,
                (unsigned long)prof->alloc_total, (unsigned long)prof->peak_usage,
                (unsigned long)prof->current_usage);
}
