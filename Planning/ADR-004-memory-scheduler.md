# ADR-004: Memory Scheduler

**Status:** Accepted
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

This is the core innovation of TemperEngine. PyTorch says "fit it in VRAM or die." We say: the scheduler figures it out.

The memory scheduler is not an allocator. It is the **brain** of the engine. Every tensor allocation, access, and deallocation routes through it. It decides:
- Where a tensor should live
- When to evict it
- Whether to recompute it instead of storing it
- How to compress it for lower tiers

Without the scheduler, TemperEngine is just another tensor library. With it, we can train models larger than GPU memory on consumer hardware.

## Decision

Multi-tier memory model with scoring-based placement and recomputation-aware eviction.

### Memory tiers

```
Tier 0: GPU VRAM       (fastest, smallest — ~8-24 GB on Apple Silicon)
Tier 1: CPU RAM        (medium speed, medium size — ~8-64 GB)
Tier 2: Compressed RAM (quantized, 2-4x density)
Tier 3: SSD            (slow, huge — ~256 GB-2 TB)
Tier 4: Recomputable   (deleted, can be regenerated)
```

On Apple Silicon, Tier 0 and Tier 1 are the same physical memory (unified memory). The distinction matters for logical placement: GPU-resident tensors are optimized for compute, CPU-resident for host operations.

### Scheduler state

```c
typedef struct TemperMemScheduler {
    // Per-tier tracking
    struct {
        size_t budget;        // max bytes allowed
        size_t used;          // current usage
        size_t reserved;      // pinned bytes (cannot evict)
    } tiers[5];

    // Resource table
    TemperResource *resources;
    uint32_t resource_count;
    uint32_t resource_capacity;

    // Configuration
    float recompute_threshold;   // score above which recomputation wins
    float pressure_low;          // below this = relaxed (0.5)
    float pressure_high;         // above this = aggressive eviction (0.85)

    // Statistics
    uint64_t total_evictions;
    uint64_t total_recomputations;
    uint64_t total_compressions;
} TemperMemScheduler;
```

### Placement scoring

When a tensor is created or requested on a device, the scheduler computes a placement score to decide the optimal tier:

```c
float temper_placement_score(TemperResource *r, TemperDevice device) {
    float score = 0.0f;

    // Factor 1: Access frequency (hot tensors prefer fast tiers)
    //   Higher access frequency = higher score for fast tiers
    float access_rate = (float)r->access_count / (float)(r->lifetime + 1);
    score += access_rate * 0.3f;

    // Factor 2: Recomputation cost (cheap to recompute = prefer eviction)
    //   Lower recompute cost = lower score (good eviction candidate)
    if (r->recomputable && r->origin) {
        float recompute_cost = (float)r->origin->recompute_flops;
        float memory_saved = (float)r->bytes;
        score -= (recompute_cost / memory_saved) * 0.2f;
    }

    // Factor 3: Copy cost (expensive to move = stay put)
    if (!temper_device_equal(r->device, device)) {
        score -= 0.2f;  // penalty for device transfer
    }

    // Factor 4: Device pressure (full device = lower score)
    float pressure = (float)tiers[device_tier(device)].used /
                     (float)tiers[device_tier(device)].budget;
    score -= pressure * 0.3f;

    return score;
}
```

### Eviction algorithm

```c
int temper_evict(TemperDevice target, size_t bytes_needed) {
    uint8_t tier = device_tier(target);

    // 1. Already have enough space?
    if (tiers[tier].used + bytes_needed <= tiers[tier].budget)
        return 0;

    // 2. Sort non-pinned resources by eviction score
    //    eviction_score = access_recency * recompute_savings
    //    low score = good eviction candidate
    TemperResource **candidates = get_eviction_candidates(tier);

    // 3. Evict until enough space freed
    for (uint32_t i = 0; tiers[tier].used + bytes_needed > tiers[tier].budget; i++) {
        TemperResource *victim = candidates[i];
        if (!victim) break;  // nothing left to evict

        if (victim->pinned) continue;  // skip pinned

        if (victim->recomputable && temper_recompute_score(victim) > scheduler->recompute_threshold) {
            // Tier 4: free it, mark for recomputation
            temper_resource_free(victim);
            victim->tier = TIER_RECOMPUTABLE;
            scheduler->total_recomputations++;
        } else if (tier > 0) {
            // Demote to next lower tier
            temper_resource_demote(victim, tier - 1);
        } else {
            // Already at lowest tier, force free
            temper_resource_free(victim);
        }
    }

    return 0;
}
```

### Resource lifecycle

```
Created → Tier 0/1 (GPU/CPU based on first op)
  ↓
Accessed frequently → stays in current tier
  ↓
Not needed temporarily → demote or recompute
  ↓
Needed again → promote back
  ↓
Training complete → free or checkpoint to SSD
```

### Pressure monitoring

```c
float temper_scheduler_pressure(TemperDevice device) {
    uint8_t tier = device_tier(device);
    return (float)tiers[tier].used / (float)tiers[tier].budget;
}

bool temper_scheduler_under_pressure(TemperDevice device) {
    return temper_scheduler_pressure(device) > scheduler->pressure_high;
}
```

## Consequences

### Enables
- Training models larger than GPU memory
- Automatic memory management (users never think about it)
- Gradient checkpointing (recompute vs. store decisions)
- Compressed tier for cold tensors
- SSD paging for very large models

### Constrains
- All tensor access routes through scheduler (one function call overhead)
- Scheduler must be initialized before any tensor operations
- Pinned tensors reduce available eviction candidates

### Tradeoffs
- **Scheduler overhead vs. user simplicity:** Every tensor access has one extra function call. Negligible for compute-bound ops. Worth it for the intelligence gained.
- **Complexity vs. capability:** The scheduler is complex. But it's the engine's differentiator. Without it, we're just another tensor library.

## Alternatives Considered

### Option A: Simple LRU eviction

**Pros:** Simple to implement.
**Cons:** Ignores recomputation possibility. No pressure awareness. No intelligence.

### Option B: User-managed placement

```c
TemperTensor a = temper_tensor_create_on(shape, TEMPER_DEVICE_GPU);
```

**Pros:** Full control.
**Cons:** Defeats the purpose. Users shouldn't think about memory management.

### Option C: OS virtual memory

**Pros:** Already exists, no work needed.
**Cons:** Too coarse-grained. OS doesn't know about tensors, tiers, or recomputation.

**Chosen: Multi-tier scoring scheduler** — This is the brain. This is what makes TemperEngine different.

## Implementation Notes

### Init

```c
void temper_scheduler_init(void) {
    // Set default budgets based on available hardware
    scheduler.tiers[TIER_GPU].budget = temper_get_gpu_memory();
    scheduler.tiers[TIER_CPU].budget = temper_get_cpu_memory();
    scheduler.tiers[TIER_COMPRESSED].budget = scheduler.tiers[TIER_CPU].budget * 2;
    scheduler.tiers[TIER_SSD].budget = temper_get_disk_free("/tmp");

    scheduler.recompute_threshold = 0.5f;
    scheduler.pressure_low = 0.5f;
    scheduler.pressure_high = 0.85f;
}
```

### Resource tracking API

```c
TemperResource *temper_scheduler_register(float *data, size_t bytes, TemperDevice device);
void temper_scheduler_unregister(TemperResource *r);
void temper_scheduler_access(TemperResource *r);  // update last_access
void temper_scheduler_pin(TemperResource *r);
void temper_scheduler_unpin(TemperResource *r);
```

## Performance Mitigations

### Per-Device Schedulers

Separate scheduler instance per device. No global lock:

```c
typedef struct TemperDeviceScheduler {
    TemperDevice device;
    size_t budget;
    size_t used;
    size_t reserved;
    TemperResource **resources;      // resources on this device
    uint32_t resource_count;
    pthread_mutex_t lock;            // per-device lock, not global
} TemperDeviceScheduler;

typedef struct TemperMemScheduler {
    TemperDeviceScheduler deviceSchedulers[TEMPER_MAX_DEVICES];
    // ... shared state ...
} TemperMemScheduler;
```

GPU scheduler and CPU scheduler run independently. Thread A can schedule GPU ops while Thread B schedules CPU ops without contention.

### Sticky Placement

If a tensor is already on a device, keep it there unless the scheduler has a strong reason to move it:

```c
TemperDevice temper_scheduler_place(TemperResource *r, TemperOpType op) {
    // Sticky: if tensor is already on a suitable device, stay there
    if (temper_op_supported_on(op, r->device) &&
        !temper_scheduler_under_pressure(r->device)) {
        return r->device;  // stay put
    }

    // Only move if necessary
    return temper_find_best_device(op, r);
}
```

This prevents ping-pong copies between CPU and GPU.

### Async Scheduling

Schedule the next op while the current one executes:

```c
typedef struct TemperAsyncScheduler {
    TemperOpQueue pending;           // ops waiting to be scheduled
    TemperOpQueue executing;         // ops currently in flight
    pthread_t scheduler_thread;      // background scheduling thread
} TemperAsyncScheduler;

void temper_async_schedule(TemperAsyncScheduler *async, TemperOp *op) {
    temper_queue_push(&async->pending, op);
}

// Background thread picks up pending ops and dispatches them
void *scheduler_thread_fn(void *arg) {
    TemperAsyncScheduler *async = arg;
    while (running) {
        TemperOp *op = temper_queue_pop(&async->pending);
        if (op) {
            temper_dispatch_op(op->type, op->inputs, op->input_count, op->output);
            temper_queue_push(&async->executing, op);
        }
    }
    return NULL;
}
```

### Batch Scheduling

Schedule the whole graph in one pass instead of per-op:

```c
int temper_schedule_graph(TemperGraph *graph) {
    // 1. Analyze all ops at once
    TemperGraphAnalysis analysis = temper_analyze_graph(graph);

    // 2. Make placement decisions for all ops
    for (uint32_t i = 0; i < graph->node_count; i++) {
        TemperGraphNode *node = graph->nodes[i];
        node->target_device = temper_place_from_analysis(node, &analysis);
    }

    // 3. Execute in order (scheduler decisions already made)
    for (uint32_t i = 0; i < graph->node_count; i++) {
        TemperGraphNode *node = graph->nodes[i];
        TemperRuntime *rt = temper_get_runtime(node->target_device);
        rt->dispatch(node->op, node->inputs, node->input_count, &node->output);
    }

    return 0;
}
```

### Decision Caching

Cache placement decisions for repeated patterns:

```c
typedef struct TemperDecisionCache {
    struct {
        TemperOpType op;
        TemperDeviceType input_devices[4];
        uint32_t input_count;
        TemperDevice result;
    } entries[1024];
    uint32_t count;
} TemperDecisionCache;

TemperDevice temper_cached_place(TemperOpType op, TemperDevice *inputs, uint32_t count) {
    // Check cache
    for (uint32_t i = 0; i < s_decision_cache.count; i++) {
        TemperDecisionCacheEntry *e = &s_decision_cache.entries[i];
        if (e->op == op && e->input_count == count) {
            bool match = true;
            for (uint32_t j = 0; j < count; j++) {
                if (!temper_device_equal(e->input_devices[j], inputs[j])) {
                    match = false;
                    break;
                }
            }
            if (match) return e->result;
        }
    }

    // Cache miss — compute and store
    TemperDevice result = temper_compute_placement(op, inputs, count);
    if (s_decision_cache.count < 1024) {
        TemperDecisionCacheEntry *e = &s_decision_cache.entries[s_decision_cache.count++];
        e->op = op;
        e->input_count = count;
        for (uint32_t j = 0; j < count; j++) e->input_devices[j] = inputs[j];
        e->result = result;
    }
    return result;
}
```

### Telemetry & Self-Tuning

Track scheduler decisions and adapt thresholds:

```c
typedef struct TemperTelemetry {
    uint64_t total_dispatches;
    uint64_t scheduler_bypasses;      // fast path hits
    uint64_t device_copies;           // tensor migrations
    uint64_t recomputations;          // tensors recomputed
    uint64_t evictions;               // tensors evicted
    uint64_t cache_hits;              // decision cache hits
    uint64_t cache_misses;            // decision cache misses
    float avg_dispatch_time_ns;
    float avg_scheduler_time_ns;
    float avg_pressure[TIER_COUNT];   // average pressure per tier
} TemperTelemetry;

// Self-tuning: adjust thresholds from real data
void temper_scheduler_tune(TemperMemScheduler *sched) {
    float bypass_rate = (float)s_telemetry.scheduler_bypasses / s_telemetry.total_dispatches;

    // If bypass rate is high (>90%), scheduler is doing well
    // If bypass rate is low (<50%), maybe thresholds are too aggressive
    if (bypass_rate < 0.5f) {
        sched->pressure_high *= 0.9f;  // relax pressure threshold
    } else if (bypass_rate > 0.95f) {
        sched->pressure_high *= 1.05f;  // tighten threshold
    }

    // Adjust recompute threshold based on recomputation frequency
    float recompute_rate = (float)s_telemetry.recomputations / s_telemetry.total_dispatches;
    if (recompute_rate > 0.1f) {
        sched->recompute_threshold *= 1.1f;  // less aggressive recomputation
    }
}
```

## Amendments (Phase 3 Implementation)

The following amendments supersede the original ADR text where they conflict. They were adopted during Phase 3 to sharpen the policy/mechanism/backend separation.

### Amendment 1: Recomputable is a state, not a tier

The tier enum is **four physical locations** only:

```
Tier 0: GPU VRAM
Tier 1: CPU RAM
Tier 2: Compressed RAM
Tier 3: SSD
```

"Recomputable" is not a location — it is a **resource state** (`TEMPER_RESOURCE_RECOMPUTABLE`). A recompute-evicted tensor is `resident == false` with **no storage at all**; it exists only as a graph node. This matters later when tensors exist purely as graph nodes. All resource state lives in a packed `uint32_t flags` (ADR-001):

```
TEMPER_RESOURCE_RESIDENT      — data is present in memory (host buffer or compressed blob)
TEMPER_RESOURCE_COMPRESSED    — data stored as backend-owned compressed blob (implies RESIDENT)
TEMPER_RESOURCE_RECOMPUTABLE  — can be regenerated from origin op
TEMPER_RESOURCE_PINNED        — scheduler cannot evict
```

### Amendment 2: Policy / Mechanism / Backend layers

Three clean layers, no leakage:

| Layer | Owns | Example |
|-------|------|---------|
| Policy | *decisions* | `temper_placement_score`, `temper_recompute_score`, thresholds |
| Mechanism | *movement* | eviction, promotion/demotion, pressure accounting, tier budgets |
| Backend | *bytes* | `TemperCompressionBackend` (bf16 in Phase 3) |

The scheduler never knows *how* a compression backend works. It only knows:

```
Can I compress this? → compress → pointer + compressed size
```

Compression metadata (`scale`, zero-point, block tables, dictionaries, checksums) lives inside an opaque `compressed_blob` owned by the backend — `TemperResource` is never extended per-algorithm.

### Amendment 3: Demote direction fix

The original eviction sample called `temper_resource_demote(victim, tier - 1)`. That moves **up** the ladder (GPU → nothing). Demotion targets `tier + 1` (GPU → CPU → COMPRESSED → SSD).

### Amendment 4: Resident vs logical bytes

Each tier tracks **both**:

```c
used     // resident footprint (compressed_size if compressed, else bytes)
logical  // uncompressed bytes of everything the graph holds in this tier
```

Recomputation reduces `used` but not `logical`. The gap (`logical - used`) measures how much of the computation graph exists without resident storage — useful for profiling.

### Amendment 5: Determinism + observability

- The scheduler is **deterministic**: identical resources, access history, and budgets produce identical decisions. LRU ties break on resource `id`. Access-frequency factors use `access_count`, not wall-clock time.
- `temper_scheduler_validate()` walks every registered resource and asserts accounting/flags/buffers are consistent. Every scheduler unit test ends with it.
- `temper_scheduler_dump_state()` prints per-tier used/logical/budget/reserved and all statistics.
- Scheduler state carries `TEMPER_SCHEDULER_VERSION` for future serialization/comparison.

### Amendment 6: Future — Memory Scheduler vs Tensor Lifetime Manager

The scheduler today owns both *where tensors live* (placement/movement) and *whether tensors should exist* (liveness). These are related but distinct problems; a Tensor Lifetime Manager may be split out in a later phase. Not built in Phase 3 — recorded here to keep the boundary in mind.

### Amendment 7: What Phase 3 defers

Per-device schedulers, async scheduling, batch scheduling, decision caching, and telemetry-driven self-tuning are **Phase 12** work. Recompute replay/cascade, gradient checkpointing, SSD paging, and advanced compression backends (int8/int4/fp8) are **Phase 7** work. Phase 3 builds the skeleton — tier model, pressure, eviction, promotion, pinning, statistics — on the CPU/GPU/COMPRESSED tiers only.
