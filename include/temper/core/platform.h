#ifndef TEMPER_PLATFORM_H
#define TEMPER_PLATFORM_H

#include <stdint.h>

typedef enum TemperPlatform
{
    TEMPER_PLATFORM_WINDOWS,
    TEMPER_PLATFORM_LINUX,
    TEMPER_PLATFORM_MACOS,
    TEMPER_PLATFORM_UNKNOWN
} TemperPlatform;

TemperPlatform temper_platform_get(void);
const char *temper_platform_name(TemperPlatform platform);
uint64_t temper_time_us(void);
void temper_sleep_ms(uint32_t ms);

#endif
