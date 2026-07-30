#include "temper/core/device.h"
#include "temper/core/logger.h"
#include "temper/core/platform.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>

static TemperDeviceTable g_device_table = {0};
static bool g_table_initialized = false;

void temper_device_table_init(void)
{
    if (g_table_initialized)
    {
        return;
    }
    memset(&g_device_table, 0, sizeof(TemperDeviceTable));

    // Register CPU_0 by default
    TemperDeviceCaps cpu_caps = {0};
    cpu_caps.supports_fp16 = false;
    cpu_caps.supports_bf16 = false;
    cpu_caps.supports_int8 = true;
    cpu_caps.total_memory = (size_t)4 * 1024 * 1024 * 1024; // Default 4GB baseline
    cpu_caps.compute_threads = 4;
    cpu_caps.tflops = 0.1f;

    g_device_table.devices[0] = TEMPER_DEVICE_CPU_0;
    g_device_table.caps[0] = cpu_caps;
    g_device_table.count = 1;

    g_table_initialized = true;
    temper_info("Device table initialized: registered default CPU_0");
}

int temper_device_register(TemperDevice device, TemperDeviceCaps caps)
{
    temper_device_table_init();
    if (g_device_table.count >= TEMPER_MAX_DEVICES)
    {
        temper_error("Failed to register device: device table full");
        return -1;
    }
    for (uint32_t i = 0; i < g_device_table.count; i++)
    {
        if (temper_device_equal(g_device_table.devices[i], device))
        {
            g_device_table.caps[i] = caps;
            return (int)i;
        }
    }
    uint32_t idx = g_device_table.count++;
    g_device_table.devices[idx] = device;
    g_device_table.caps[idx] = caps;
    temper_info("Registered device %s (id: %u)", temper_device_name(device), device.id);
    return (int)idx;
}

uint32_t temper_device_count(void)
{
    temper_device_table_init();
    return g_device_table.count;
}

const TemperDeviceTable *temper_device_table_get(void)
{
    temper_device_table_init();
    return &g_device_table;
}

bool temper_device_equal(TemperDevice a, TemperDevice b)
{
    return a.type == b.type && a.id == b.id;
}

bool temper_device_is_cpu(TemperDevice device)
{
    return device.type == TEMPER_DEVICE_CPU;
}

bool temper_device_is_gpu(TemperDevice device)
{
    return device.type == TEMPER_DEVICE_GPU;
}

const char *temper_device_name(TemperDevice device)
{
    static char buf[32];
    const char *type_str = "Unknown";
    switch (device.type)
    {
    case TEMPER_DEVICE_CPU:
        type_str = "CPU";
        break;
    case TEMPER_DEVICE_GPU:
        type_str = "GPU";
        break;
    case TEMPER_DEVICE_NPU:
        type_str = "NPU";
        break;
    }
    snprintf(buf, sizeof(buf), "%s:%u", type_str, device.id);
    return buf;
}

TemperDeviceCaps temper_device_get_caps(TemperDevice device)
{
    temper_device_table_init();
    for (uint32_t i = 0; i < g_device_table.count; i++)
    {
        if (temper_device_equal(g_device_table.devices[i], device))
        {
            return g_device_table.caps[i];
        }
    }
    TemperDeviceCaps default_caps = {0};
    return default_caps;
}
