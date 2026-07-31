#include "temper/core/runtime.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/metal/metal.h"
#include <stdlib.h>
#include <string.h>

static TemperRuntime g_cpu_runtime;
static TemperRuntime *g_runtimes[TEMPER_MAX_RUNTIMES];
static uint32_t g_runtime_count = 0;
static bool g_initialized = false;

static void *cpu_alloc_host(size_t bytes)
{
    return calloc(1, bytes);
}

static void cpu_free_host(void *ptr)
{
    free(ptr);
}

static int cpu_init(void)
{
    return 0;
}

static void cpu_shutdown(void)
{
}

static void cpu_synchronize(void)
{
}

static void cpu_wait_idle(void)
{
}

void temper_runtime_table_init(void)
{
    if (g_initialized)
    {
        return;
    }

    memset(g_runtimes, 0, sizeof(g_runtimes));
    g_runtime_count = 0;

    g_cpu_runtime = (TemperRuntime){
        .name = "cpu",
        .device = TEMPER_DEVICE_CPU_0,
        .alloc = cpu_alloc_host,
        .free = cpu_free_host,
        .alloc_host = cpu_alloc_host,
        .free_host = cpu_free_host,
        .init = cpu_init,
        .shutdown = cpu_shutdown,
        .synchronize = cpu_synchronize,
        .wait_idle = cpu_wait_idle,
    };
    g_runtimes[g_runtime_count++] = &g_cpu_runtime;
    g_initialized = true;

    temper_info("Runtime table initialized: registered %s", g_cpu_runtime.name);
}

int temper_runtime_register(TemperRuntime *runtime)
{
    if (!runtime || !runtime->name || !runtime->alloc_host || !runtime->free_host)
    {
        temper_error("Failed to register runtime: invalid runtime");
        return -1;
    }
    temper_runtime_table_init();

    for (uint32_t i = 0; i < g_runtime_count; i++)
    {
        if (temper_device_equal(g_runtimes[i]->device, runtime->device))
        {
            temper_info("Replaced runtime %s with %s for device %s",
                        g_runtimes[i]->name, runtime->name,
                        temper_device_name(runtime->device));
            g_runtimes[i] = runtime;
            return 0;
        }
    }

    if (g_runtime_count >= TEMPER_MAX_RUNTIMES)
    {
        temper_error("Failed to register runtime %s: runtime table full", runtime->name);
        return -1;
    }
    g_runtimes[g_runtime_count++] = runtime;
    temper_info("Registered runtime %s for device %s", runtime->name,
                temper_device_name(runtime->device));
    return 0;
}

uint32_t temper_runtime_count(void)
{
    temper_runtime_table_init();
    return g_runtime_count;
}

TemperRuntime *temper_get_runtime(TemperDevice device)
{
    temper_runtime_table_init();
    for (uint32_t i = 0; i < g_runtime_count; i++)
    {
        if (temper_device_equal(g_runtimes[i]->device, device))
        {
            return g_runtimes[i];
        }
    }
    return NULL;
}

TemperRuntime *temper_get_runtime_by_type(TemperDeviceType type)
{
    temper_runtime_table_init();
    for (uint32_t i = 0; i < g_runtime_count; i++)
    {
        if (g_runtimes[i]->device.type == type)
        {
            return g_runtimes[i];
        }
    }
    return NULL;
}

TemperRuntime *temper_cpu_runtime(void)
{
    temper_runtime_table_init();
    return &g_cpu_runtime;
}

TemperRuntime *temper_runtime_ensure(TemperDevice device)
{
    temper_runtime_table_init();
    TemperRuntime *rt = temper_get_runtime(device);
    if (rt)
    {
        return rt;
    }
    // Lazy backend initialization on first use (PERFORMANCE.md #19): only touch
    // platform runtimes when a device they can serve is actually requested.
    if (device.type == TEMPER_DEVICE_GPU && temper_platform_get() == TEMPER_PLATFORM_MACOS)
    {
        if (temper_metal_maybe_init() == 0)
        {
            return temper_get_runtime(device);
        }
    }
    return NULL;
}

void temper_runtime_shutdown_all(void)
{
    if (!g_initialized)
    {
        return;
    }
    for (uint32_t i = 0; i < g_runtime_count; i++)
    {
        if (g_runtimes[i]->shutdown)
        {
            g_runtimes[i]->shutdown();
        }
        temper_info("Runtime %s shut down", g_runtimes[i]->name);
    }
    g_runtime_count = 0;
    g_initialized = false;
}
