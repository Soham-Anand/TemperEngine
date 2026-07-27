#include "temper/core/logger.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static TemperLogLevel s_level = TEMPER_LOG_TRACE;

static const char *level_strings[] = {"TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};

void temper_log_set_level(TemperLogLevel level)
{
    s_level = level;
}

void temper_log(TemperLogLevel level, const char *file, int line, const char *fmt, ...)
{
    if (level < s_level)
    {
        return;
    }

    const char *base = strrchr(file, '/');
    if (!base)
    {
        base = strrchr(file, '\\');
    }
    if (!base)
    {
        base = file;
    }
    else
    {
        base++;
    }

    fprintf(stderr, "[%s] %s:%d: ", level_strings[level], base, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}
