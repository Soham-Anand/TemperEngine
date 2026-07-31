#include "temper/temper.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/core/runtime.h"
#include "temper/compute/kernel.h"
#include <stdio.h>

static int s_initialized = 0;

int temper_init(void)
{
    if (s_initialized)
    {
        return 0;
    }

    temper_log_set_level(TEMPER_LOG_TRACE);

    TemperPlatform platform = temper_platform_get();
    temper_info("TemperEngine %s initializing on %s", TEMPER_VERSION_STRING,
                temper_platform_name(platform));
    temper_info("sizeof(void*) = %zu", sizeof(void *));

    temper_runtime_table_init();
    temper_kernel_init();

    s_initialized = 1;
    return 0;
}

void temper_shutdown(void)
{
    if (!s_initialized)
    {
        return;
    }
    temper_runtime_shutdown_all();
    temper_info("TemperEngine shutdown");
    s_initialized = 0;
}
