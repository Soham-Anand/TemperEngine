# ADR-002: Device Abstraction

**Status:** Accepted
**Date:** 2026-07-27
**Accepted:** 2026-07-31 (Phase 4 — device table, caps, GPU registration verified on Apple M4)
**Deciders:** Soham Anand

## Context

We need to represent compute devices without hardcoding "GPU." Apple Silicon has GPU + Neural Engine. Desktop has discrete GPU + CPU. Future has multiple GPUs, TPUs, NPUs.

The current `TEMPER_DEVICE_CPU` / `TEMPER_DEVICE_GPU` enum is too simple:
- Cannot distinguish between multiple GPUs
- Cannot represent NPU or future accelerators
- Cannot query device capabilities
- Cannot target a specific device by ID

## Decision

Two-part device identity: type enum + device ID, plus a capabilities structure.

### Device type + ID

```c
typedef enum TemperDeviceType {
    TEMPER_DEVICE_CPU,
    TEMPER_DEVICE_GPU,       // Metal, CUDA, Vulkan — runtime-specific
    TEMPER_DEVICE_NPU,       // Apple Neural Engine, future accelerators
} TemperDeviceType;

typedef struct TemperDevice {
    TemperDeviceType type;
    uint32_t id;             // 0 = primary, 1 = secondary, etc.
} TemperDevice;

// Built-in constants
#define TEMPER_DEVICE_CPU_0  ((TemperDevice){TEMPER_DEVICE_CPU, 0})
#define TEMPER_DEVICE_GPU_0  ((TemperDevice){TEMPER_DEVICE_GPU, 0})
#define TEMPER_DEVICE_NPU_0  ((TemperDevice){TEMPER_DEVICE_NPU, 0})
```

### Device capabilities

```c
typedef struct TemperDeviceCaps {
    bool supports_fp16;
    bool supports_bf16;
    bool supports_int8;
    size_t total_memory;     // bytes
    size_t compute_threads;  // CPU cores or GPU compute units
    float tflops;            // peak theoretical throughput
} TemperDeviceCaps;
```

Each runtime reports its device capabilities at initialization. The memory scheduler uses these for placement decisions.

### Device table

```c
#define TEMPER_MAX_DEVICES 16

typedef struct TemperDeviceTable {
    TemperDevice devices[TEMPER_MAX_DEVICES];
    TemperDeviceCaps caps[TEMPER_MAX_DEVICES];
    uint32_t count;
} TemperDeviceTable;
```

The engine maintains a global device table. Runtimes register their devices at init time.

### Helper functions

```c
bool temper_device_equal(TemperDevice a, TemperDevice b);
bool temper_device_is_cpu(TemperDevice device);
bool temper_device_is_gpu(TemperDevice device);
const char *temper_device_name(TemperDevice device);
TemperDeviceCaps temper_device_get_caps(TemperDevice device);
```

## Consequences

### Enables
- Multi-GPU: `GPU_0`, `GPU_1`, etc.
- Apple Neural Engine: `NPU_0`
- Future accelerators without API changes
- Scheduler queries capabilities to decide placement
- User code never sees device IDs — the scheduler picks

### Constrains
- Device comparison requires two-field check (type + ID)
- Device table has a fixed maximum (16 devices — sufficient for foreseeable future)

### Tradeoffs
- **Fixed array vs. dynamic list:** Fixed is simpler, no allocation. 16 devices is enough for any single machine.
- **Enum vs. string:** Enum is fast for comparison (hot path), string is more flexible. Enum wins.

## Alternatives Considered

### Option A: Single `TEMPER_DEVICE_GPU` enum

```c
typedef enum TemperDevice {
    TEMPER_DEVICE_CPU,
    TEMPER_DEVICE_GPU,
} TemperDevice;
```

**Pros:** Simple.
**Cons:** Cannot distinguish multi-GPU or GPU vs NPU. Dead end for heterogeneous compute.

### Option B: String-based device names

```c
typedef struct TemperDevice {
    const char *name;  // "metal:0", "cuda:1", "npu:0"
} TemperDevice;
```

**Pros:** Fully flexible, no enum maintenance.
**Cons:** Slow string comparison in hot paths. Error-prone. No type safety.

### Option C: Capability flags only

```c
typedef struct TemperDevice {
    uint32_t flags;  // TEMPER_CAP_FP16 | TEMPER_CAP_GPU | ...
} TemperDevice;
```

**Pros:** Very flexible.
**Cons:** No identity — cannot target a specific device. Can't say "put this on GPU 1."

**Chosen: Type + ID** — Provides identity (for targeting) and type (for capability queries). Simple, fast, extensible.

## Implementation Notes

```c
// Register a device (called by runtime init)
void temper_device_register(TemperDevice device, TemperDeviceCaps caps);

// Find devices matching criteria
TemperDevice temper_device_find(TemperDeviceType type, uint32_t min_memory);

// Get all GPUs
uint32_t temper_device_get_gpus(TemperDevice *out, uint32_t max_count);
```

## Implementation Status

Implemented in `include/temper/core/device.h` / `src/core/device.c`.

- Two-part identity (`TemperDeviceType` + `uint32_t id`) and `TemperDeviceCaps` are
  implemented as specified, with `TEMPER_DEVICE_CPU_0` / `TEMPER_DEVICE_GPU_0` /
  `TEMPER_DEVICE_NPU_0` constants.
- The device table registers the default `CPU:0` at init; the Metal runtime registers
  `GPU:0` with real capabilities (M4: fp16, bf16, int8 support, memory and compute
  unit counts from the `MTLDevice`) when it initializes.
- `temper_device_register` replaces a same-id device entry (idempotent re-init);
  `temper_device_find`/`temper_device_get_gpus` round out the query surface.
- GPU capability query (`temper_device_get_caps`) is verified by
  `tests/test_metal_runtime.c`; device table behavior by `tests/test_runtime.c`.
