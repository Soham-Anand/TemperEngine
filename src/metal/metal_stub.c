#include "temper/metal/metal.h"

// Stub for non-Apple platforms: keeps metal.h symbols linkable so core code
// (runtime registry, examples) builds everywhere without #ifdef sprawl.

int temper_metal_maybe_init(void)
{
    return -1;
}

bool temper_metal_is_available(void)
{
    return false;
}

void temper_metal_shutdown(void)
{
}
