# ADR-001: Tensor Representation

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

The tensor is the fundamental data structure of TemperEngine. Every operation — forward pass, backward pass, optimizer step — operates on tensors. The way we represent tensors determines:

- How memory is managed across devices (CPU, GPU, NPU)
- How the memory scheduler can track, evict, and recompute tensors
- How operations dispatch to the correct runtime
- How autograd tracks the computation graph

The current implementation stores a raw `float *data` pointer directly in the tensor struct. This works for single-device CPU-only execution but cannot support:

- Multi-device tensor placement
- Memory scheduler control over allocation
- Backend-specific buffer types (MTLBuffer, cudaMalloc, VkBuffer)
- Recomputation and eviction policies

## Decision

We adopt a **two-layer representation**: a lightweight `TemperTensor` handle that holds metadata, and an indirect `TemperResource` pointer that holds the actual data and device-specific state.

### TemperResource

```c
typedef struct TemperResource {
    uint32_t id;                  // unique resource ID for scheduler tracking
    TemperDevice device;          // where this resource lives
    void *native;                 // backend-specific handle (MTLBuffer, raw pointer, etc.)
    float *host_ptr;              // CPU-accessible pointer (may be NULL if GPU-only)
    size_t bytes;                 // total allocation size in bytes
    uint64_t last_access;         // timestamp of last read/write
    uint32_t refcount;            // how many tensors reference this resource
    bool pinned;                  // if true, scheduler cannot evict
    bool recomputable;            // if true, can be regenerated from inputs
    TemperGraphNode *origin;      // if recomputable, the op that produced this
} TemperResource;
```

### TemperTensor (handle)

```c
typedef struct TemperTensor {
    TemperResource *resource;     // indirect pointer to data
    TemperShape shape;            // logical shape
    TemperDType dtype;            // data type (f32, f16, bf16, i8, u8)
    int64_t stride[TEMPER_MAX_DIMS]; // stride per dimension (enables views)
    uint32_t refcount;            // tensor-level refcount (multiple views can share a resource)
} TemperTensor;
```

### Key design choices

1. **Indirection through TemperResource.** The tensor never owns memory directly. It points to a resource that the memory scheduler owns and tracks. This allows the scheduler to evict, replace, or recompute the resource without modifying the tensor.

2. **Resource ID for tracking.** Every resource gets a unique ID. The memory scheduler maintains a table of all resources and their states. This enables LRU tracking, access pattern analysis, and recomputation decisions.

3. **Recomputable flag.** If a tensor's resource is marked recomputable, the scheduler can free the memory and regenerate it later by replaying the origin op. This is the foundation of gradient checkpointing.

4. **Stride-based views.** `transpose`, `slice`, and `reshape` create new tensor handles that share the same resource but with different stride/shape. This avoids copying data for view operations.

5. **Pinned tensors.** Optimizer state, current weights during backward, and user-pinned tensors cannot be evicted. The scheduler respects pinning.

## Consequences

### Enables
- Memory scheduler can evict/replace resources transparently
- Multi-device placement (same tensor handle, different resource per device)
- Gradient checkpointing (recompute vs. store decisions)
- Stride-based views without data copies
- Reference counting for safe deallocation

### Constrains
- Every tensor access requires one pointer indirection (resource->native)
- The memory scheduler must be initialized before any tensor operations
- View tensors share a resource — modifying one view affects all views sharing it

### Tradeoffs
- **Performance:** One indirection per access. Mitigated by the scheduler keeping hot resources in cache-friendly locations. The overhead is negligible compared to compute.
- **Complexity:** Two-level indirection is harder to reason about than raw pointers. Mitigated by clear ownership rules: scheduler owns resources, user code owns tensors.

## Alternatives Considered

### Option A: Raw pointer in tensor (current approach)

```c
typedef struct TemperTensor {
    float *data;
    TemperShape shape;
    // ...
} TemperTensor;
```

**Pros:** Simple, no indirection, direct memory access.
**Cons:** Cannot support multi-device, no scheduler control, no eviction, no recomputation. Dead end.

### Option B: Reference-counted tensor handle

```c
typedef struct TemperTensor {
    struct TemperTensor *base;   // if this is a view, points to the base tensor
    float *data;
    // ...
} TemperTensor;
```

**Pros:** Views are cheap. Familiar pattern (NumPy, PyTorch).
**Cons:** Still stores raw pointer. No resource tracking. Scheduler cannot intercept allocation.

### Option C: Capability-based tensor (like JAX)

```c
typedef struct TemperTensor {
    void *data;
    TemperDevice device;
    TemperDType dtype;
    TemperShape shape;
    // capabilities: what ops can run on this tensor
} TemperTensor;
```

**Pros:** Tensor knows what it can do.
**Cons:** Over-engineered. Dispatch should be the scheduler's job, not the tensor's.

**Chosen: Option A (TemperResource indirection)** — The scheduler needs to own memory decisions. The tensor is a handle. The resource is the thing.

## Performance Mitigations

### Hot/Cold Resource Split

Split `TemperResource` into hot and cold fields for cache efficiency:

```c
// Hot path (fits in cache line — 64 bytes)
typedef struct TemperResourceHot {
    uint32_t id;              // 4 bytes
    TemperDevice device;      // 4 bytes (type + id)
    float *host_ptr;          // 8 bytes
    void *native;             // 8 bytes
    uint32_t flags;           // 4 bytes (packed: pinned, recomputable, dirty, ...)
    size_t bytes;             // 8 bytes
    // Total: 36 bytes — fits in cache line
} TemperResourceHot;

// Cold path (rarely accessed, separate allocation)
typedef struct TemperResourceCold {
    uint64_t last_access;     // for LRU tracking
    uint32_t refcount;
    uint32_t access_count;    // for recomputation scoring
    uint64_t lifetime;        // ticks since creation
    TemperGraphNode *origin;  // for recomputation
    char debug_name[32];      // optional debug label
} TemperResourceCold;
```

The scheduler operates on hot fields. Cold fields are accessed only during eviction decisions and debugging.

### Tensor Pooling

Reuse tensor structs for small, short-lived tensors:

```c
typedef struct TemperTensorPool {
    TemperTensor *free_list;       // linked list of free tensor handles
    uint32_t free_count;
    uint32_t capacity;
} TemperTensorPool;

TemperTensor *temper_tensor_pool_acquire(TemperTensorPool *pool);
void temper_tensor_pool_release(TemperTensorPool *pool, TemperTensor *t);
```

Tensor pools eliminate malloc/free overhead for the millions of temporary tensors created during a training step.

### Slab Allocation

Allocate tensors of the same size from contiguous blocks:

```c
typedef struct TemperSlab {
    char *buffer;              // contiguous block
    size_t block_size;         // size of each block
    size_t block_count;        // number of blocks
    uint32_t *free_bitmap;    // bitmap of free blocks
} TemperSlab;

void *temper_slab_alloc(TemperSlab *slab);
void temper_slab_free(TemperSlab *slab, void *ptr);
```

Slabs reduce fragmentation and improve cache locality for tensors of similar size.

### Inline Scalars

Store scalar values directly in the tensor struct. Skip heap allocation:

```c
typedef struct TemperTensor {
    TemperResource *resource;     // NULL for inline scalars
    TemperShape shape;
    TemperDType dtype;
    int64_t stride[TEMPER_MAX_DIMS];
    uint32_t refcount;
    // Inline storage for scalars (0-d tensors)
    float scalar_value;
    bool is_scalar;
} TemperTensor;
```

A `temper_tensor_create_scalar(3.14f)` never touches the heap.

### Bitfield Flags

Pack boolean flags into a single uint32_t:

```c
// Instead of:
//   bool pinned;
//   bool recomputable;
//   bool dirty;
//   bool compressed;
//   bool is_view;

// Use:
uint32_t flags;
#define TEMPER_FLAG_PINNED        (1 << 0)
#define TEMPER_FLAG_RECOMPUTABLE  (1 << 1)
#define TEMPER_FLAG_DIRTY         (1 << 2)
#define TEMPER_FLAG_COMPRESSED    (1 << 3)
#define TEMPER_FLAG_IS_VIEW       (1 << 4)

// Access:
bool is_pinned = (resource->flags & TEMPER_FLAG_PINNED) != 0;
resource->flags |= TEMPER_FLAG_DIRTY;
resource->flags &= ~TEMPER_FLAG_DIRTY;
```

Saves 3 bytes per resource. More importantly, a single cache line fetch gives access to all flags.

## Implementation Notes

```c
// Creating a tensor (scheduler allocates resource)
TemperTensor temper_tensor_create(TemperShape shape, TemperDType dtype);

// Creating a scalar (inline, no heap allocation)
TemperTensor temper_tensor_create_scalar(float value);

// Moving a tensor to a different device (scheduler creates new resource)
TemperTensor temper_tensor_to(const TemperTensor *t, TemperDevice device);

// Accessing data (must ensure resource is resident)
float *temper_tensor_data(const TemperTensor *t);  // triggers promotion if needed

// Creating a view (shares resource, different shape/stride)
TemperTensor temper_tensor_view(const TemperTensor *t, TemperShape new_shape,
                                int64_t *new_stride);

// Pooling (reuse tensor handles)
TemperTensor *temper_tensor_acquire(void);
void temper_tensor_release(TemperTensor *t);
```
