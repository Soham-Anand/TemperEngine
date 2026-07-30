#ifndef TEMPER_DEVICE_H
#define TEMPER_DEVICE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum TemperDeviceType
{
    TEMPER_DEVICE_CPU,
    TEMPER_DEVICE_GPU,
    TEMPER_DEVICE_NPU
} TemperDeviceType;

typedef struct TemperDevice
{
    TemperDeviceType type;
    uint32_t id;
} TemperDevice;

#define TEMPER_DEVICE_CPU_0 ((TemperDevice){TEMPER_DEVICE_CPU, 0})
#define TEMPER_DEVICE_GPU_0 ((TemperDevice){TEMPER_DEVICE_GPU, 0})
#define TEMPER_DEVICE_NPU_0 ((TemperDevice){TEMPER_DEVICE_NPU, 0})

typedef struct TemperDeviceCaps
{
    bool supports_fp16;
    bool supports_bf16;
    bool supports_int8;
    size_t total_memory;     // Total device memory in bytes
    size_t compute_threads;  // CPU cores or GPU compute units
    float tflops;            // Theoretical peak TFLOPS
} TemperDeviceCaps;

#define TEMPER_MAX_DEVICES 16

typedef struct TemperDeviceTable
{
    TemperDevice devices[TEMPER_MAX_DEVICES];
    TemperDeviceCaps caps[TEMPER_MAX_DEVICES];
    uint32_t count;
} TemperDeviceTable;

void temper_device_table_init(void);
int temper_device_register(TemperDevice device, TemperDeviceCaps caps);
uint32_t temper_device_count(void);
const TemperDeviceTable *temper_device_table_get(void);

bool temper_device_equal(TemperDevice a, TemperDevice b);
bool temper_device_is_cpu(TemperDevice device);
bool temper_device_is_gpu(TemperDevice device);
const char *temper_device_name(TemperDevice device);
TemperDeviceCaps temper_device_get_caps(TemperDevice device);

#endif
