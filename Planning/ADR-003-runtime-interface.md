# ADR-003: Runtime Interface

**Status:** Accepted
**Date:** 2026-07-27
**Accepted:** 2026-07-31 (Phase 4 — CPU + Metal runtimes live; Metal verified on Apple M4)
**Deciders:** Soham Anand

## Context

Backends (CPU, Metal, CUDA) need a uniform way to plug into the engine. The engine dispatches operations through the runtime, and the runtime executes them on its device.

We need:
- A consistent interface for all backends
- Runtime registration at engine init
- Query by device to get the right runtime
- Graceful fallback if a runtime fails

## Decision

`TemperRuntime` — a function pointer table that each backend implements.

### Runtime struct

```c
typedef struct TemperRuntime {
    const char *name;
    TemperDevice device;

    // Memory
    void *(*alloc)(size_t bytes);
    void *(*alloc_host)(size_t bytes);     // CPU-accessible (unified memory on Apple Silicon)
    void  (*free)(void *ptr);
    void  (*free_host)(void *ptr);

    // Compute
    int (*dispatch)(TemperOpType op,
                    const TemperTensor *const *inputs,
                    uint32_t input_count,
                    TemperTensor *output);

    // Synchronization
    void (*synchronize)(void);
    void (*wait_idle)(void);

    // Lifecycle
    int  (*init)(void);
    void (*shutdown)(void);
} TemperRuntime;
```

### Registration

```c
#define TEMPER_MAX_RUNTIMES 8

typedef struct TemperRuntimeTable {
    TemperRuntime runtimes[TEMPER_MAX_RUNTIMES];
    uint32_t count;
} TemperRuntimeTable;

// Global runtime table
static TemperRuntimeTable s_runtimes = {0};

// Called by each backend at init
void temper_register_runtime(TemperRuntime runtime) {
    if (s_runtimes.count >= TEMPER_MAX_RUNTIMES) return;
    s_runtimes.runtimes[s_runtimes.count++] = runtime;
}

// Called by scheduler to get runtime for a device
TemperRuntime *temper_get_runtime(TemperDevice device) {
    for (uint32_t i = 0; i < s_runtimes.count; i++) {
        if (temper_device_equal(s_runtimes.runtimes[i].device, device)) {
            return &s_runtimes.runtimes[i];
        }
    }
    return NULL;  // no runtime for this device
}
```

### CPU runtime (always available)

```c
TemperRuntime temper_cpu_runtime_create(void) {
    TemperRuntime rt = {0};
    rt.name = "cpu";
    rt.device = TEMPER_DEVICE_CPU_0;
    rt.alloc = cpu_alloc;
    rt.alloc_host = cpu_alloc;       // same as alloc on CPU
    rt.free = cpu_free;
    rt.free_host = cpu_free;
    rt.dispatch = cpu_dispatch;
    rt.synchronize = cpu_synchronize;  // no-op on CPU
    rt.wait_idle = cpu_synchronize;
    rt.init = cpu_init;
    rt.shutdown = cpu_shutdown;
    return rt;
}
```

### Metal runtime (macOS only)

```c
TemperRuntime temper_metal_runtime_create(void) {
    TemperRuntime rt = {0};
    rt.name = "metal";
    rt.device = TEMPER_DEVICE_GPU_0;
    rt.alloc = metal_alloc;          // MTLBuffer allocation
    rt.alloc_host = metal_alloc_host; // unified memory on Apple Silicon
    rt.free = metal_free;
    rt.free_host = metal_free;
    rt.dispatch = metal_dispatch;    // Metal compute pipeline
    rt.synchronize = metal_synchronize;
    rt.wait_idle = metal_wait_idle;
    rt.init = metal_init;            // create MTLDevice, command queue
    rt.shutdown = metal_shutdown;
    return rt;
}
```

## Consequences

### Enables
- Each backend implements one `TemperRuntime` struct
- Scheduler queries `temper_get_runtime(device)` to dispatch
- Runtimes are interchangeable — swap Metal for CUDA without changing user code
- CPU runtime is always registered as fallback
- Multiple runtimes can coexist (CPU + Metal + future CUDA)

### Constrains
- All runtimes must implement the full interface (even if some are no-ops)
- Runtime dispatch is a function pointer call (one level of indirection)
- Runtime init/shutdown must be called in correct order

### Tradeoffs
- **Function pointers vs. switch statement:** Function pointers allow runtime addition without modifying core code. Switch requires recompilation.
- **Fixed table vs. dynamic list:** Fixed is simpler, no allocation. 8 runtimes is enough.

## Alternatives Considered

### Option A: Plugin system (dlopen)

**Pros:** Third-party backends without recompilation.
**Cons:** Complex, platform-specific, security concerns. Not suitable for v1.

### Option B: Compile-time backend list

```c
#ifdef TEMPER_HAS_METAL
    temper_register_runtime(temper_metal_runtime_create());
#endif
```

**Pros:** No runtime overhead.
**Cons:** Inflexible, can't add backends without recompilation.

### Option C: Inheritance (C++)

**Pros:** Clean polymorphism.
**Cons:** Project is C17, not C++. Would require language change.

**Chosen: Static function pointer table** — Simple, extensible, no external dependencies. Plugin system can be added later on top of this.

## Performance Mitigations

### Runtime Capability Cache

Cache which ops each runtime supports. Avoid repeated capability queries:

```c
typedef struct TemperRuntimeCaps {
    TemperDevice device;
    bool op_supported[TEMPER_OP_COUNT];
    size_t max_tensor_bytes;
    float dispatch_overhead_ns;   // measured dispatch cost
} TemperRuntimeCaps;

// Populated at init time
static TemperRuntimeCaps s_caps[TEMPER_MAX_RUNTIMES];

bool temper_op_supported_on(TemperOpType op, TemperDevice device) {
    for (uint32_t i = 0; i < s_runtimes.count; i++) {
        if (temper_device_equal(s_caps[i].device, device)) {
            return s_caps[i].op_supported[op];
        }
    }
    return false;
}
```

### Fast-Path Bypass

Skip the scheduler entirely for the common case:

```c
int temper_dispatch_op(TemperOpType op,
                       const TemperTensor *const *inputs,
                       uint32_t input_count,
                       TemperTensor *output) {
    // Fast path: same device, no pressure, op supported
    TemperDevice dev = inputs[0]->resource->device;
    bool all_same = true;
    for (uint32_t i = 1; i < input_count; i++) {
        if (!temper_device_equal(inputs[i]->resource->device, dev)) {
            all_same = false;
            break;
        }
    }

    if (all_same &&
        !temper_scheduler_under_pressure(dev) &&
        temper_op_supported_on(op, dev)) {
        // Skip scheduler — direct dispatch
        TemperRuntime *rt = temper_get_runtime(dev);
        return rt->dispatch(op, inputs, input_count, output);
    }

    // Slow path: go through scheduler
    return temper_dispatch_slow(op, inputs, input_count, output);
}
```

Most ops hit the fast path. Only cross-device or pressured situations go through the scheduler.

### Runtime Pointer Caching

Cache the last-used runtime per device type. Most ops use the same device consecutively:

```c
static TemperRuntime *s_cached_runtimes[TEMPER_DEVICE_TYPE_COUNT] = {0};

TemperRuntime *temper_get_runtime(TemperDevice device) {
    // Check cache first
    if (s_cached_runtimes[device.type] &&
        temper_device_equal(s_cached_runtimes[device.type]->device, device)) {
        return s_cached_runtimes[device.type];
    }

    // Cache miss — search table
    for (uint32_t i = 0; i < s_runtimes.count; i++) {
        if (temper_device_equal(s_runtimes.runtimes[i].device, device)) {
            s_cached_runtimes[device.type] = &s_runtimes.runtimes[i];
            return s_cached_runtimes[device.type];
        }
    }
    return NULL;
}
```

### Lazy Runtime Initialization

Only initialize a runtime when it's first used:

```c
typedef struct TemperRuntime {
    // ... existing fields ...
    bool initialized;
} TemperRuntime;

int temper_ensure_initialized(TemperRuntime *rt) {
    if (!rt->initialized) {
        int ret = rt->init();
        if (ret == 0) rt->initialized = true;
        return ret;
    }
    return 0;
}

// Called before dispatch
int temper_dispatch_op(TemperOpType op, ...) {
    TemperRuntime *rt = temper_get_runtime(target);
    temper_ensure_initialized(rt);
    return rt->dispatch(op, ...);
}
```

Metal init is expensive (create MTLDevice, command queue). Lazy init avoids paying that cost if GPU is never used.

## Implementation Notes

### Dispatch flow

```c
int temper_dispatch_op(TemperOpType op,
                       const TemperTensor *const *inputs,
                       uint32_t input_count,
                       TemperTensor *output) {
    // 1. Scheduler determines target device
    TemperDevice target = temper_scheduler_place(inputs, input_count);

    // 2. Ensure inputs are on target device
    TemperTensor *local[4];
    for (uint32_t i = 0; i < input_count; i++) {
        local[i] = temper_tensor_to(inputs[i], target);
    }

    // 3. Get runtime
    TemperRuntime *rt = temper_get_runtime(target);
    if (!rt) {
        // Fallback to CPU
        rt = temper_get_runtime(TEMPER_DEVICE_CPU_0);
    }

    // 4. Dispatch
    return rt->dispatch(op, local, input_count, output);
}
```

### Init order

```c
int temper_init(void) {
    // 1. Core systems (logger, profiler, etc.)
    temper_core_init();

    // 2. CPU runtime (always)
    temper_register_runtime(temper_cpu_runtime_create());

    // 3. Platform runtimes (try Metal, CUDA, etc.)
    #ifdef TEMPER_HAS_METAL
    if (temper_metal_is_available()) {
        temper_register_runtime(temper_metal_runtime_create());
    }
    #endif

    // 4. Memory scheduler
    temper_scheduler_init();

    return 0;
}
```

## Implementation Status

Implemented in `include/temper/core/runtime.h` / `src/core/runtime.c` with one
deliberate divergence from the ADR sketch: **compute dispatch does not live in the
runtime**. `TemperRuntime` owns memory policy (`alloc_host`/`free_host`), device
lifecycle (`init`/`shutdown`), and synchronization — it has no `dispatch` entry point.
Compute dispatch lives in the kernel registry (`temper/compute/kernel.h`), which
separates **what** to compute (`TemperKernelType`) from **how** to compute it on a
device (`TemperKernelImpl`). This split was accepted in the Phase 4 design review.

### Dispatch stack

```
API op (temper_tensor_add, ...)
   |
   v
placement policy (Phase 4: stay on the tensor's device; no automatic migration yet)
   |
   v
TemperKernelType  "what"        (TEMPER_KERNEL_ADD, TEMPER_KERNEL_MATMUL, ...)
   |
   v
TemperKernelImpl  "how"         (cpu_add, metal_add, metal_matmul_tiled, ...)
   |                            selected by (type, device_type); replaceable at runtime
   v
TemperRuntime                   owns the target device's memory + sync
   |
   v
Device                          CPU_0 / GPU_0 / NPU_0
```

Telemetry flows back up the same stack: each implementation fills a
`TemperKernelReport` (time, flops, bytes read/written, occupancy), aggregated into
per-type `TemperKernelStats`. This is the seed of the measurement-driven scheduler
loop (telemetry -> placement); Phase 4 only collects, it never decides placement.

### What is implemented

- CPU runtime always registered at `temper_init`; Metal runtime registered lazily by
  `temper_runtime_ensure(device)` on the first GPU resource creation (avoids the
  expensive MTLDevice/command-queue init when the GPU is never used).
- `temper_runtime_shutdown_all` tears down registered runtimes in reverse order at
  `temper_shutdown`.
- Memory: resource create/release route through `alloc_host`/`free_host`, so the
  core sees only allocate -> pointer -> size. Metal allocates shared-memory
  `MTLBuffer`s (host_ptr -> MTLBuffer registry); an mmap-wrapped zero-copy pool is
  planned as an allocator swap, not a rewrite.
- Zero-copy `temper_tensor_to(GPU -> CPU)`: GPU tensors on unified memory are
  CPU-addressable, so the move is a metadata-only resource migrate
  (`temper_resource_migrate`), not a copy.
- Non-Apple builds link `src/metal/metal_stub.c` so `temper_runtime_ensure` stays
  a no-op and the library is portable.
- Verified by `tests/test_runtime.c` (13 tests) and `tests/test_metal_runtime.c`
  (9 tests, macOS/Metal only), green under Debug, Release, and ASan+UBSan.
