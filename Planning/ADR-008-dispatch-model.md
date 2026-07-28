# ADR-008: Dispatch Model

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

When a user writes `temper_add(a, b)`, the engine must:
1. Determine which device to run on
2. Ensure both tensors are on that device
3. Dispatch to the correct runtime
4. Return the result

This must be transparent — the user never specifies devices, never sees Metal or CUDA, never thinks about memory placement. The engine figures it all out.

## Decision

Three-layer dispatch: API → Scheduler → Runtime.

### User API (clean, no device knowledge)

```c
// User writes this — no device, no dispatch, no runtime
TemperTensor c = temper_add(a, b);
TemperTensor d = temper_matmul(c, weights);
TemperTensor e = temper_relu(d);
```

### Dispatch implementation

```c
TemperTensor temper_add(const TemperTensor *a, const TemperTensor *b) {
    // 1. Query memory scheduler for placement
    TemperDevice target = temper_scheduler_place_op(TEMPER_OP_ADD, a, b);

    // 2. Ensure inputs are on target device
    TemperTensor a_local = temper_tensor_to(a, target);
    TemperTensor b_local = temper_tensor_to(b, target);

    // 3. Get runtime for target device
    TemperRuntime *rt = temper_get_runtime(target);
    if (!rt) {
        // Fallback to CPU
        rt = temper_get_runtime(TEMPER_DEVICE_CPU_0);
        target = TEMPER_DEVICE_CPU_0;
    }

    // 4. Allocate output resource
    TemperTensor output = temper_tensor_create_like(&a_local);
    output.resource->device = target;

    // 5. Dispatch
    int ret = rt->dispatch(TEMPER_OP_ADD,
                           (const TemperTensor *[]){&a_local, &b_local},
                           2, &output);

    // 6. Fallback on failure
    if (ret != 0 && target.type != TEMPER_DEVICE_CPU) {
        // Retry on CPU
        a_local = temper_tensor_to(a, TEMPER_DEVICE_CPU_0);
        b_local = temper_tensor_to(b, TEMPER_DEVICE_CPU_0);
        rt = temper_get_runtime(TEMPER_DEVICE_CPU_0);
        output.resource->device = TEMPER_DEVICE_CPU_0;
        rt->dispatch(TEMPER_OP_ADD,
                     (const TemperTensor *[]){&a_local, &b_local},
                     2, &output);
    }

    return output;
}
```

### Scheduler placement logic

```c
TemperDevice temper_scheduler_place_op(TemperOpType op,
                                       const TemperTensor *a,
                                       const TemperTensor *b) {
    // Rule 1: If both on same device, stay there
    if (temper_device_equal(a->resource->device, b->resource->device)) {
        return a->resource->device;
    }

    // Rule 2: If one is on GPU, promote to GPU (if not under pressure)
    TemperDevice gpu = temper_device_find_gpu();
    if (gpu.type != TEMPER_DEVICE_UNKNOWN) {
        if ((a->resource->device.type == TEMPER_DEVICE_GPU ||
             b->resource->device.type == TEMPER_DEVICE_GPU) &&
            !temper_scheduler_under_pressure(gpu)) {
            return gpu;
        }
    }

    // Rule 3: Small tensors stay on CPU (skip GPU launch overhead)
    size_t total_bytes = temper_tensor_bytes(a) + temper_tensor_bytes(b);
    if (total_bytes < 4096) {  // < 4KB
        return TEMPER_DEVICE_CPU_0;
    }

    // Rule 4: Default to CPU
    return TEMPER_DEVICE_CPU_0;
}
```

### Fallback chain

```
1. Try target device (GPU)
2. If GPU fails (OOM, unsupported op) → try CPU
3. If CPU fails → try compressed tier (slower but more capacity)
4. If all fail → error (should be rare with recomputation)
```

### Op registry

Each runtime registers which ops it supports:

```c
typedef struct TemperOpRegistry {
    TemperOpType op;
    TemperDeviceType device;
    bool supported;
} TemperOpRegistry;

// Query
bool temper_op_supported(TemperOpType op, TemperDevice device) {
    for (uint32_t i = 0; i < s_op_registry_count; i++) {
        if (s_op_registry[i].op == op &&
            s_op_registry[i].device == device.type) {
            return s_op_registry[i].supported;
        }
    }
    return false;  // unknown = not supported
}
```

### Error handling

```c
typedef enum TemperDispatchError {
    TEMPER_DISPATCH_OK = 0,
    TEMPER_DISPATCH_OOM,           // out of memory on target device
    TEMPER_DISPATCH_UNSUPPORTED,  // op not supported on device
    TEMPER_DISPATCH_FAILED,       // runtime error
} TemperDispatchError;
```

The dispatch function returns an error code. If non-zero, the caller retries on CPU.

### Device promotion

When inputs are on different devices, the scheduler promotes to the "best" device:

```c
TemperDevice temper_promote_device(TemperDevice a, TemperDevice b) {
    // GPU beats CPU
    if (a.type == TEMPER_DEVICE_GPU) return a;
    if (b.type == TEMPER_DEVICE_GPU) return b;

    // NPU beats CPU (if available)
    if (a.type == TEMPER_DEVICE_NPU) return a;
    if (b.type == TEMPER_DEVICE_NPU) return b;

    // Default to CPU
    return TEMPER_DEVICE_CPU_0;
}
```

## Consequences

### Enables
- Transparent multi-device execution
- Automatic placement (user never thinks about devices)
- Graceful fallback (GPU fails → CPU works)
- Small tensor optimization (skip GPU for tiny ops)

### Constrains
- Every op goes through scheduler (one function call overhead)
- Fallback requires retry (extra dispatch on failure)
- Op registry must be maintained per runtime

### Tradeoffs
- **Indirection vs. flexibility:** One function call per op is negligible for compute-bound ops. Worth it for transparent multi-device.

## Alternatives Considered

### Option A: User specifies device

```c
TemperTensor c = temper_add_on(a, b, TEMPER_DEVICE_GPU);
```

**Pros:** Full control.
**Cons:** Defeats the purpose. Users shouldn't think about devices.

### Option B: Compile-time dispatch

```c
#ifdef TEMPER_HAS_METAL
    temper_add_metal(a, b);
#else
    temper_add_cpu(a, b);
#endif
```

**Pros:** No runtime overhead.
**Cons:** Inflexible, can't adapt to runtime memory pressure.

### Option C: Direct runtime call (no scheduler)

```c
rt->dispatch(TEMPER_OP_ADD, inputs, 2, &output);
```

**Pros:** Fastest.
**Cons:** No placement intelligence, no fallback, no device promotion.

**Chosen: Three-layer dispatch** — API → Scheduler → Runtime. The scheduler makes intelligent decisions. The user doesn't think about it.

## Implementation Notes

### All tensor ops follow the same pattern

```c
// Macro to generate dispatch functions
#define TEMPER_DISPATCH_OP(OP_NAME, OP_TYPE)                                \
TemperTensor OP_NAME(const TemperTensor *a, const TemperTensor *b) {       \
    TemperDevice target = temper_scheduler_place_op(OP_TYPE, a, b);        \
    TemperTensor a_l = temper_tensor_to(a, target);                        \
    TemperTensor b_l = temper_tensor_to(b, target);                        \
    TemperTensor out = temper_tensor_create_like(&a_l);                    \
    TemperRuntime *rt = temper_get_runtime(target);                        \
    rt->dispatch(OP_TYPE, (const TemperTensor *[]){&a_l, &b_l}, 2, &out); \
    return out;                                                            \
}

TEMPER_DISPATCH_OP(temper_add, TEMPER_OP_ADD)
TEMPER_DISPATCH_OP(temper_sub, TEMPER_OP_SUB)
TEMPER_DISPATCH_OP(temper_mul, TEMPER_OP_MUL)
TEMPER_DISPATCH_OP(temper_div, TEMPER_OP_DIV)
```

### Op-to-kernel mapping

```c
// CPU runtime
int cpu_dispatch(TemperOpType op, const TemperTensor *const *inputs,
                 uint32_t input_count, TemperTensor *output) {
    switch (op) {
    case TEMPER_OP_ADD:  return cpu_add(inputs, input_count, output);
    case TEMPER_OP_SUB:  return cpu_sub(inputs, input_count, output);
    case TEMPER_OP_MUL:  return cpu_mul(inputs, input_count, output);
    case TEMPER_OP_DIV:  return cpu_div(inputs, input_count, output);
    case TEMPER_OP_MATMUL: return cpu_matmul(inputs, input_count, output);
    case TEMPER_OP_RELU: return cpu_relu(inputs, input_count, output);
    // ...
    default: return TEMPER_DISPATCH_UNSUPPORTED;
    }
}

// Metal runtime
int metal_dispatch(TemperOpType op, const TemperTensor *const *inputs,
                   uint32_t input_count, TemperTensor *output) {
    switch (op) {
    case TEMPER_OP_ADD:  return metal_add(inputs, input_count, output);
    case TEMPER_OP_MUL:  return metal_mul(inputs, input_count, output);
    case TEMPER_OP_RELU: return metal_relu(inputs, input_count, output);
    // ... fewer ops initially
    default: return TEMPER_DISPATCH_UNSUPPORTED;  // fallback to CPU
    }
}
```

## Performance Mitigations

### Command Batching

Group multiple dispatches into a single runtime call. Reduce overhead:

```c
typedef struct TemperCommandBatch {
    TemperOpType ops[TEMPER_MAX_BATCH_SIZE];
    const TemperTensor *inputs[TEMPER_MAX_BATCH_SIZE][4];
    uint32_t input_counts[TEMPER_MAX_BATCH_SIZE];
    TemperTensor *outputs[TEMPER_MAX_BATCH_SIZE];
    uint32_t count;
    TemperDevice device;
} TemperCommandBatch;

int temper_batch_dispatch(TemperCommandBatch *batch) {
    TemperRuntime *rt = temper_get_runtime(batch->device);
    return rt->dispatch_batch(batch->ops, batch->inputs,
                              batch->input_counts, batch->outputs,
                              batch->count);
}
```

CPU runtime can use SIMD or multi-threading to execute batched ops in parallel. Metal runtime can encode multiple kernels into a single command buffer.

### Kernel Batching (Metal-specific)

Submit multiple Metal compute kernels without synchronization between them:

```c
int metal_dispatch_batch(TemperCommandBatch *batch) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    for (uint32_t i = 0; i < batch->count; i++) {
        // Encode kernel
        [enc setComputePipelineState:pipelines[batch->ops[i]]];
        [enc setBuffer:batch->inputs[i][0]->resource->native offset:0 index:0];
        [enc setBuffer:batch->inputs[i][1]->resource->native offset:0 index:1];
        [enc setBuffer:batch->outputs[i]->resource->native offset:0 index:2];

        // Dispatch without endEncoding/startEncoding between kernels
        MTLSize gridSize = MTLSizeMake(batch->outputs[i]->shape.dims[0], 1, 1);
        MTLSize groupSize = MTLSizeMake(32, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return 0;
}
```

No sync between kernels = GPU stays busy.

### Lazy Migration

Don't copy tensors immediately. Mark them for migration and copy only when accessed:

```c
typedef enum TemperMigrationState {
    TEMPER_MIGRATION_NONE = 0,
    TEMPER_MIGRATION_PENDING,   // marked for copy, not yet done
    TEMPER_MIGRATION_IN_FLIGHT, // copy in progress
    TEMPER_MIGRATION_DONE,      // copy complete
} TemperMigrationState;

// Mark tensor for lazy migration
void temper_tensor_mark_migrate(TemperTensor *t, TemperDevice target) {
    t->resource->pending_device = target;
    t->resource->migration_state = TEMPER_MIGRATION_PENDING;
}

// Trigger actual copy when data is needed
float *temper_tensor_data(TemperTensor *t) {
    if (t->resource->migration_state == TEMPER_MIGRATION_PENDING) {
        temper_tensor_do_migrate(t);
        t->resource->migration_state = TEMPER_MIGRATION_DONE;
    }
    return t->resource->host_ptr;
}
```

Lazy migration allows the scheduler to batch multiple tensor copies together and make better decisions about which tensors to actually move.

### Small-Op Threshold

Skip GPU dispatch for tiny tensors:

```c
#define SMALL_TENSOR_BYTES 4096  // 4 KB

int temper_dispatch_op(TemperOpType op, ...) {
    size_t total_bytes = 0;
    for (uint32_t i = 0; i < input_count; i++) {
        total_bytes += inputs[i]->resource->bytes;
    }

    // Small tensors always go to CPU (skip GPU launch overhead)
    if (total_bytes < SMALL_TENSOR_BYTES) {
        TemperRuntime *rt = temper_get_runtime(TEMPER_DEVICE_CPU_0);
        return rt->dispatch(op, inputs, input_count, output);
    }

    // Normal dispatch path
    // ...
}
```

GPU kernel launch overhead is ~10-50 microseconds. For tensors smaller than 4KB, CPU compute is faster than the launch cost.

### Fallback Chain with Retry

Graceful fallback when GPU dispatch fails:

```c
int temper_dispatch_op_safe(TemperOpType op, const TemperTensor *const *inputs,
                            uint32_t input_count, TemperTensor *output) {
    // 1. Try target device
    TemperDevice target = temper_scheduler_place_op(op, inputs, input_count);
    TemperRuntime *rt = temper_get_runtime(target);
    int ret = rt->dispatch(op, inputs, input_count, output);

    if (ret == TEMPER_DISPATCH_OK) return 0;

    // 2. Try CPU
    if (target.type != TEMPER_DEVICE_CPU) {
        rt = temper_get_runtime(TEMPER_DEVICE_CPU_0);
        ret = rt->dispatch(op, inputs, input_count, output);
        if (ret == TEMPER_DISPATCH_OK) {
            temper_info("Dispatch fallback: %s → CPU", temper_op_name(op));
            return 0;
        }
    }

    // 3. All failed
    temper_error("Dispatch failed for %s on all devices", temper_op_name(op));
    return ret;
}
```
