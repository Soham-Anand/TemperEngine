#ifndef TEMPER_H
#define TEMPER_H

#define TEMPER_VERSION_MAJOR 0
#define TEMPER_VERSION_MINOR 1
#define TEMPER_VERSION_PATCH 0
#define TEMPER_VERSION_STRING "0.1.0"

#include "temper/core/memory.h"
#include "temper/core/logger.h"
#include "temper/core/profiler.h"
#include "temper/core/platform.h"

int temper_init(void);
void temper_shutdown(void);

#endif
