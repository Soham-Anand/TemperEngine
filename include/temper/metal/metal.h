#ifndef TEMPER_METAL_H
#define TEMPER_METAL_H

#include <stdbool.h>

// Metal backend entry points. The header is portable (declarations only); the
// implementation lives in metal.m and is compiled only on Apple platforms.
// temper_metal_maybe_init() is called lazily by the runtime registry the first
// time a GPU device is requested, so a machine without Metal (or a binary built
// on Linux/Windows) pays no cost.

bool temper_metal_is_available(void);
int temper_metal_maybe_init(void);
void temper_metal_shutdown(void);

#endif
