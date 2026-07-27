#include "temper/core/platform.h"

#ifdef __APPLE__
    #include <mach/mach_time.h>
    #include <unistd.h>
    #define TEMPEN_PLATFORM_ID TEMPER_PLATFORM_MACOS
#elif defined(_WIN32)
    #include <windows.h>
    #define TEMPEN_PLATFORM_ID TEMPER_PLATFORM_WINDOWS
#else
    #include <time.h>
    #include <unistd.h>
    #define TEMPEN_PLATFORM_ID TEMPER_PLATFORM_LINUX
#endif

TemperPlatform temper_platform_get(void)
{
    return TEMPEN_PLATFORM_ID;
}

const char *temper_platform_name(TemperPlatform platform)
{
    switch (platform)
    {
    case TEMPER_PLATFORM_WINDOWS:
        return "Windows";
    case TEMPER_PLATFORM_LINUX:
        return "Linux";
    case TEMPER_PLATFORM_MACOS:
        return "macOS";
    default:
        return "Unknown";
    }
}

uint64_t temper_time_us(void)
{
#ifdef __APPLE__
    return mach_absolute_time() / 1000;
#elif defined(_WIN32)
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000 / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
#endif
}

void temper_sleep_ms(uint32_t ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    usleep((useconds_t)ms * 1000);
#endif
}
