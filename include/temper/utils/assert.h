#ifndef TEMPER_ASSERT_H
#define TEMPER_ASSERT_H

#include "temper/core/logger.h"

#ifdef NDEBUG
    #define TEMPER_ASSERT(cond) ((void)0)
    #define TEMPER_ASSERT_MSG(cond, msg) ((void)0)
#else
    #define TEMPER_ASSERT(cond)                                                 \
        do                                                                      \
        {                                                                       \
            if (!(cond))                                                        \
            {                                                                   \
                temper_fatal("Assertion failed: %s at %s:%d", #cond, __FILE__, \
                             __LINE__);                                         \
                __builtin_trap();                                               \
            }                                                                   \
        } while (0)

    #define TEMPER_ASSERT_MSG(cond, msg)                                        \
        do                                                                      \
        {                                                                       \
            if (!(cond))                                                        \
            {                                                                   \
                temper_fatal("Assertion failed: %s — %s at %s:%d", #cond, msg, \
                             __FILE__, __LINE__);                               \
                __builtin_trap();                                               \
            }                                                                   \
        } while (0)
#endif

#endif
