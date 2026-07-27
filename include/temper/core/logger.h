#ifndef TEMPER_LOGGER_H
#define TEMPER_LOGGER_H

typedef enum TemperLogLevel
{
    TEMPER_LOG_TRACE,
    TEMPER_LOG_DEBUG,
    TEMPER_LOG_INFO,
    TEMPER_LOG_WARN,
    TEMPER_LOG_ERROR,
    TEMPER_LOG_FATAL
} TemperLogLevel;

void temper_log_set_level(TemperLogLevel level);
void temper_log(TemperLogLevel level, const char *file, int line, const char *fmt, ...);

#define temper_trace(...) temper_log(TEMPER_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define temper_debug(...) temper_log(TEMPER_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define temper_info(...)  temper_log(TEMPER_LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define temper_warn(...)  temper_log(TEMPER_LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define temper_error(...) temper_log(TEMPER_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define temper_fatal(...) temper_log(TEMPER_LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

#endif
