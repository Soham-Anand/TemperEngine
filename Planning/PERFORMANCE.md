# TemperEngine — Performance & Mitigations Reference

This document catalogs all known architectural consequences and their mitigations. It is a reference document, not an architectural decision record.

---

## Table of Contents

1. [Scheduler Bottleneck](#1-scheduler-bottleneck)
2. [Device Transfer Overhead](#2-device-transfer-overhead)
3. [Recomputation Explosion](#3-recomputation-explosion)
4. [Runtime Dispatch Overhead](#4-runtime-dispatch-overhead)
5. [Metadata Overhead](#5-metadata-overhead)
6. [Dynamic Graph Overhead](#6-dynamic-graph-overhead)
7. [Backend Maintenance](#7-backend-maintenance)
8. [Testing Explosion](#8-testing-explosion)
9. [Serialization Compatibility](#9-serialization-compatibility)
10. [Resource Lookup Bottleneck](#10-resource-lookup-bottleneck)
11. [CPU Cache Misses](#11-cpu-cache-misses)
12. [Thread Contention](#12-thread-contention)
13. [Memory Fragmentation](#13-memory-fragmentation)
14. [Graph Optimization](#14-graph-optimization)
15. [Small Operation Overhead](#15-small-operation-overhead)
16. [Poor GPU Utilization](#16-poor-gpu-utilization)
17. [CPU Underutilization](#17-cpu-underutilization)
18. [Scheduler Mistakes](#18-scheduler-mistakes)
19. [Slow Startup](#19-slow-startup)
20. [Debugging Complexity](#20-debugging-complexity)

---

## 1. Scheduler Bottleneck

**Severity:** ⭐⭐⭐⭐⭐

### Problem

Every tensor operation goes through the scheduler. Millions of operations = millions of scheduler calls.

### Mitigations

**Fast-path bypass:**
```c
if (a.device == b.device &&
    !scheduler_under_pressure &&
    op_is_supported) {
    // Skip scheduler entirely — direct dispatch
}
```

**Batch scheduling:**
Instead of scheduling each op individually, schedule the whole graph in one pass.

**Decision caching:**
Cache placement decisions for repeated patterns. If `(MatMul, GPU)` was chosen once, reuse that decision for similar tensors.

---

## 2. Device Transfer Overhead

**Severity:** ⭐⭐⭐⭐⭐

### Problem

CPU → GPU → CPU → GPU copy chains kill performance.

### Mitigations

**Sticky placement:**
If a tensor is already on GPU, keep it there unless the scheduler has a strong reason to move it.

**Predict future usage:**
If a tensor will be used 100 more times, leave it on GPU. Don't move it back to CPU.

**Lazy migration:**
Don't copy immediately. Mark tensor as `NEEDS_GPU`. Copy only when it's actually accessed on GPU.

---

## 3. Recomputation Explosion

**Severity:** ⭐⭐⭐⭐☆

### Problem

Cascading recomputation (A needs B needs C needs D) can blow up compute.

### Mitigations

**Maximum recomputation depth:**
```c
#define MAX_RECOMPUTE_DEPTH 5
```
If depth > 5, store the tensor instead of recomputing.

**Adaptive scoring:**
If a tensor keeps getting recomputed (>10 times), store it permanently.

**Learning scheduler:**
Track recomputation counts per tensor. After training, tensors that were recomputed frequently get pinned.

---

## 4. Runtime Dispatch Overhead

**Severity:** ⭐⭐⭐☆☆

### Problem

Function pointer lookup every operation.

### Mitigations

**Runtime pointer caching:**
Cache the last-used runtime per device. Most ops use the same device consecutively.

**Command batching:**
Group multiple dispatches into a single runtime call.

---

## 5. Metadata Overhead

**Severity:** ⭐⭐⭐☆☆

### Problem

Small tensors have huge metadata relative to their data.

### Mitigations

**Inline scalars:**
Store scalar values directly in the tensor struct. Skip heap allocation.

**Bitfield flags:**
```c
// Instead of:
bool pinned;
bool compressed;
bool dirty;

// Use:
uint32_t flags;  // bit 0 = pinned, bit 1 = compressed, bit 2 = dirty
```

---

## 6. Dynamic Graph Overhead

**Severity:** ⭐⭐⭐⭐☆

### Problem

Every operation allocates graph nodes.

### Mitigations

**Arena allocator:**
Allocate all graph nodes from an arena. Reset the arena per training step. No malloc/free overhead.

**Object pooling:**
Reuse graph nodes from a free list instead of allocating new ones.

**Gradient tensor pooling:**
Reuse gradient tensors of the same shape instead of allocating new ones.

---

## 7. Backend Maintenance

**Severity:** ⭐⭐⭐⭐☆

### Problem

Need CPU, Metal, CUDA, Vulkan implementations.

### Mitigations

**Shared kernel definitions:**
Define element-wise ops as templates. Generate CPU, Metal, CUDA code from the same source.

**Common runtime utilities:**
Don't duplicate buffer creation, synchronization, error handling across backends.

---

## 8. Testing Explosion

**Severity:** ⭐⭐⭐⭐☆

### Problem

Hundreds of device/op/dtype combinations.

### Mitigations

**Parameterized tests:**
Run the same test across CPU, GPU, Mixed, Compressed automatically.

**Property-based testing:**
Generate random tensors. Compare all runtimes produce identical results.

---

## 9. Serialization Compatibility

**Severity:** ⭐⭐⭐☆☆

### Problem

Can't break `.temper` format across versions.

### Mitigations

**Version adapters:**
Load old format → convert to internal format → use normally.

**Unknown section skipping:**
Forward-compatible: skip sections the loader doesn't recognize.

---

## 10. Resource Lookup Bottleneck

**Severity:** ⭐⭐⭐⭐☆

### Problem

Linear search through resource table is O(n).

### Mitigations

**Dense array indexed by ID:**
```c
TemperResource *resources_by_id[MAX_RESOURCES];  // O(1) lookup
```

**Hash map fallback:**
For sparse IDs, use a hash map with resource ID as key.

---

## 11. CPU Cache Misses

**Severity:** ⭐⭐⭐⭐☆

### Problem

Pointer chasing: Tensor → Resource → Data causes cache misses.

### Mitigations

**Hot/cold split:**
```c
// Hot path (fits in cache line — 64 bytes)
struct TemperResourceHot {
    uint32_t id;
    TemperDevice device;
    float *host_ptr;
    void *native;
    uint32_t flags;
};

// Cold path (rarely accessed)
struct TemperResourceCold {
    uint64_t last_access;
    uint32_t refcount;
    TemperGraphNode *origin;
    char debug_name[32];
};
```

---

## 12. Thread Contention

**Severity:** ⭐⭐⭐⭐☆

### Problem

Global scheduler lock serializes all operations.

### Mitigations

**Per-device schedulers:**
Separate scheduler instance per device. GPU scheduler and CPU scheduler run independently.

**Lock-free queues:**
Use lock-free SPSC queues for inter-thread communication.

**Thread-local allocations:**
Allocate from thread-local pools to avoid contention.

---

## 13. Memory Fragmentation

**Severity:** ⭐⭐⭐⭐☆

### Problem

Thousands of small allocations fragment memory.

### Mitigations

**Arena allocator:**
Bulk allocate, bulk free.

**Tensor pool:**
Pre-allocate pools of common tensor sizes.

**Slab allocator:**
Allocate tensors of the same size from contiguous blocks.

**Huge pages (optional):**
Use OS huge pages for large allocations to reduce TLB misses.

---

## 14. Graph Optimization

**Severity:** ⭐⭐⭐⭐☆

### Problem

Executing unnecessary ops (e.g., transpose of transpose).

### Mitigations

See ADR-009: Graph Optimization Pipeline.

- Dead tensor elimination
- Constant folding
- Identity elimination
- Operator fusion
- Buffer lifetime planning
- Kernel batching

---

## 15. Small Operation Overhead

**Severity:** ⭐⭐⭐☆☆

### Problem

Scheduler costs more than computation for tiny tensors.

Example: `add` on 8 numbers — scheduler overhead > compute time.

### Mitigations

**Small-op threshold:**
```c
#define SMALL_TENSOR_BYTES 4096  // 4 KB

if (tensor_bytes < SMALL_TENSOR_BYTES) {
    // Always CPU, skip scheduler
}
```

---

## 16. Poor GPU Utilization

**Severity:** ⭐⭐⭐⭐⭐

### Problem

Many tiny kernels with gaps between them.

### Mitigations

**Kernel fusion:**
Fuse consecutive element-wise ops into one kernel launch.

**Command batching:**
Submit multiple kernels without synchronization between them.

**Persistent kernels:**
Keep GPU busy with long-running kernels that process multiple batches.

---

## 17. CPU Underutilization

**Severity:** ⭐⭐⭐⭐☆

### Problem

Only GPU working while CPU sits idle.

### Mitigations

**Real heterogeneous scheduling:**
```
GPU: MatMul
CPU: LayerNorm + Optimizer + Data loading
```
Both compute simultaneously.

**Pipeline parallelism:**
CPU preloads batch N+1 while GPU trains batch N.

---

## 18. Scheduler Mistakes

**Severity:** ⭐⭐⭐⭐⭐

### Problem

Bad heuristics lead to poor placement decisions.

### Mitigations

**Telemetry:**
Track copies, recomputes, cache hits, evictions, runtime per decision.

**Self-tuning scheduler:**
Instead of fixed `4 KB` threshold, learn the best threshold per machine from telemetry data.

---

## 19. Slow Startup

**Severity:** ⭐⭐⭐☆☆

### Problem

Initializing all subsystems takes time.

### Mitigations

**Lazy initialization:**
Only initialize Metal when GPU is first used. Only initialize CUDA if NVIDIA GPU detected.

**Subsystems:**
```
Core (logger, profiler): always init
CPU runtime: always init
Metal runtime: init on first GPU op
CUDA runtime: init on first CUDA op
Memory scheduler: init on first tensor creation
```

---

## 20. Debugging Complexity

**Severity:** ⭐⭐⭐⭐☆

### Problem

Many layers make debugging hard.

### Mitigations

**Built-in inspector:**
```c
temper_inspect(tensor);
// → Location: GPU_0
// → History: [CPU → GPU copy at step 142]
// → Owner: forward pass, layer 3
// → Graph node: #847
```

**Timeline profiler:**
```
Tensor moved    [0.02ms]
Scheduler       [0.01ms]
Dispatch        [0.03ms]
Kernel          [1.20ms]
```

**Pass statistics (from ADR-009):**
Show what the optimizer did, how much it saved, how long it took.

---

## Priority Matrix

| Mitigation | Impact | Effort | Priority |
|------------|--------|--------|----------|
| Fast-path bypass | High | Low | P0 |
| Per-device schedulers | High | Medium | P0 |
| Hot/cold resource split | High | Medium | P0 |
| Tensor pooling | Medium | Low | P1 |
| Inline scalars | Medium | Low | P1 |
| Arena graph allocation | High | Medium | P1 |
| Small-op threshold | Medium | Low | P1 |
| Kernel fusion | High | High | P2 |
| Command batching | Medium | Medium | P2 |
| Lazy migration | Medium | Medium | P2 |
| Telemetry/self-tuning | High | High | P2 |
| Recomputation depth limit | High | Low | P1 |
| Decision caching | Medium | Low | P1 |
| Buffer lifetime planning | High | High | P2 |
| Shared kernel definitions | Medium | High | P3 |
| Property-based testing | Medium | Medium | P3 |
| Built-in inspector | Medium | Medium | P3 |

---

## Phase 4 — Metal Backend Results (measured 2026-07-31, Apple M4)

Status of §2/§16/§18/§19 mitigations at the end of Phase 4.

### Implemented so far

- **§19 Lazy initialization:** the Metal runtime is created only on the first GPU
  resource creation (`temper_runtime_ensure`); the MTLDevice + command queue are
  never touched in CPU-only workloads.
- **§2 Device transfer:** GPU tensors are Metal **shared-memory** buffers, so a
  GPU→CPU `temper_tensor_to` is a metadata-only resource migrate, not a copy.
  Buffers are allocated through the runtime's `alloc_host` contract, so an
  mmap + `newBufferWithBytesNoCopy` pool stays a drop-in allocator swap.
- **§18 Telemetry:** every kernel implementation fills a `TemperKernelReport`
  (time, flops, bytes read/written, occupancy) aggregated per kernel type
  (`TemperKernelStats`). Phase 4 collects only; the telemetry→placement loop is
  Phase 5/12.
- **§16 Kernel quality:** one custom tiled matmul (16×16 tiles, shared memory)
  replaces the naive triple loop. Kernel fusion and command batching are still
  open (P2) — the element-wise path is one-command-buffer-per-op.

### Measured numbers (`benchmarks/bench_kernel`)

| M x K x N | CPU ms | GPU ms | Speedup | GFLOPS | GB/s |
|-----------|--------|--------|---------|--------|------|
| 16 x 16 x 16 | 0.01 | 0.17 | 0.04x | 0.05 | 0.02 |
| 64 x 64 x 64 | 0.33 | 0.18 | 1.84x | 2.88 | 0.27 |
| 256 x 256 x 256 | 24.38 | 0.66 | 37.17x | 51.15 | 1.20 |
| 1024 x 1024 x 1024 | 3139.48 | 13.81 | 227.40x | 155.55 | 0.91 |
| 4096 x 4096 x 4096 | n/a* | 444.68 | — | 309.07 | 0.45 |
| 128 x 768 x 3072 | 884.22 | 4.44 | 199.15x | 136.03 | 2.57 |
| 1024 x 4096 x 1024 | n/a* | 27.31 | — | 314.53 | 1.38 |
| 512 x 64 x 2048 | 151.40 | 1.64 | 92.15x | 81.69 | 2.95 |

Element-wise add: 16M elements → 100.56x vs CPU, 28.7 GB/s.

\* CPU column skipped above an 8 GFLOP budget (naive triple-loop reference).

### Notes / next steps

- Tiled kernel reaches ~310 GFLOPS (~12% of M4 peak fp32). Bigger tiles,
  SIMD-group accumulators, and a second tiled variant for tall-skinny shapes are
  the natural Phase-5 matmul work; the benchmark harness makes each change
  measurable before and after.
- Small-shape overhead (16³ and the 0.17 ms floor) is dispatch + pipeline fixed
  cost — §15 (small-op threshold) and §16 (batching) directly target it.
- Scheduler profiling was explicitly deferred (maintainer decision 2026-07-31);
  run a scheduler-focused profile pass before Phase 5 wires placement into
  dispatch.

---

*This document is a living reference. Add new mitigations as they are discovered during implementation and profiling.*
