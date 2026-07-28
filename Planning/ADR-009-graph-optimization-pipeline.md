# ADR-009: Graph Optimization & Compilation Pipeline

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

The graph optimization pipeline sits between graph capture and memory scheduling. It transforms the user's computation graph into an optimized form before the scheduler makes placement decisions.

Without optimization, the scheduler receives a raw graph with:
- Dead tensors (outputs never used)
- Redundant operations (transpose of transpose)
- Unnecessary kernel launches (two element-wise ops that could be one)
- No memory reuse information
- No lifetime analysis

The compiler cleans this up, giving the scheduler better information and reducing total work.

## Decision

A multi-phase compilation pipeline with analysis, canonicalization, optimization passes, verification, and pass statistics.

### Pipeline position

```
User API
      │
      ▼
Graph Capture
      │
      ▼
Verification (Pre-Opt)
      │
      ▼
Canonicalization
      │
      ▼
Analysis Layer
      │
      ▼
Optimization Passes (topological order)
      │
      ▼
Verification (Post-Opt)
      │
      ▼
Memory Scheduler
      │
      ▼
Runtime Dispatch
      │
      ▼
CPU / Metal / CUDA / ...
```

### Pass Manager

```c
#define TEMPER_MAX_DEPS 8

typedef struct TemperPass {
    const char *name;
    uint32_t id;
    uint32_t dependencies[TEMPER_MAX_DEPS];  // must run before this pass
    uint32_t dep_count;
    int (*run)(TemperGraph *graph, TemperAnalysis *analysis);
    TemperPassStats stats;
    bool enabled;
} TemperPass;

typedef struct TemperPassManager {
    TemperPass *passes;
    uint32_t pass_count;
    uint32_t capacity;
    TemperOptLevel opt_level;
} TemperPassManager;
```

### Pass Statistics

Every pass returns statistics for profiling and tuning:

```c
typedef struct TemperPassStats {
    uint32_t nodes_removed;
    uint32_t nodes_added;
    uint32_t ops_fused;
    uint32_t ops_simplified;
    size_t memory_saved_bytes;
    uint64_t runtime_ns;
} TemperPassStats;
```

Displayed by the profiler:

```
Constant Folding
  Removed: 128 nodes
  Saved:   12 MB
  Time:    0.21 ms

Operator Fusion
  Fused:   64 pairs → 32 kernels
  Saved:   8 MB
  Time:    0.15 ms
```

### Optimization Levels

```c
typedef enum TemperOptLevel {
    TEMPER_O0 = 0,  // No optimization — debug, fast compile
    TEMPER_O1 = 1,  // Basic — dead elimination, constant folding, identity elimination
    TEMPER_O2 = 2,  // Balanced — + fusion, buffer planning
    TEMPER_O3 = 3,  // Aggressive — + multi-pass, speculative fusion, kernel batching
} TemperOptLevel;
```

| Level | Passes | Use Case |
|-------|--------|----------|
| O0 | None | Debugging, fast iteration |
| O1 | Dead elim, constant fold, identity elim | Quick optimization |
| O2 | + Fusion, buffer planning | Production training |
| O3 | + Multi-pass, speculative fusion | Maximum performance |

### Verification (runs twice)

```c
// Pre-optimization: catches invalid user graphs
int temper_verify_graph_pre(TemperGraph *graph);

// Post-optimization: catches optimizer bugs
int temper_verify_graph_post(TemperGraph *graph);
```

Checks:
- Shape consistency (all dims match)
- Device consistency (no mixed-device ops without copy)
- Gradient availability (all trainable tensors have gradient path)
- Resource integrity (all tensors point to valid resources)
- No cycles in the graph

### Graph Canonicalization

Before optimization, normalize equivalent forms:

```c
int temper_canonicalize(TemperGraph *graph);
```

Rules:
- **Commutative ops:** `A + B` and `B + A` → canonical order (smaller ID first)
- **Associative chains:** `(A + B) + C` → flat chain `A + B + C`
- **Transpose cancellation:** `transpose(transpose(A))` → `A`
- **Reshape fusion:** `reshape(reshape(A, s1), s2)` → `reshape(A, s2)`
- **Identity ops:** `add(A, 0)` → `A`, `mul(A, 1)` → `A`

Canonicalization makes later passes simpler and more predictable.

### Analysis Layer

Before optimization, compute analysis results that passes consume:

```c
typedef struct TemperAnalysis {
    // Liveness: which tensors are live at each point
    TemperLiveness *liveness;

    // Dependencies: op dependency graph
    TemperDependencyGraph *deps;

    // Alias analysis: which tensors share memory
    TemperAliasMap *aliases;

    // Shape propagation: inferred shapes for all nodes
    TemperShapeMap *shapes;

    // Device requirements: which devices each op supports
    TemperDeviceMap *devices;

    // Lifetime: when each tensor is first/last used
    TemperLifetimeMap *lifetime;
} TemperAnalysis;
```

Analysis is computed once. All optimization passes consume it. Passes do NOT recompute analysis — they request fresh analysis if they invalidate it.

### Optimization Passes

Each pass implements:

```c
typedef struct TemperPassImpl {
    const char *name;
    uint32_t id;
    uint32_t dependencies[TEMPER_MAX_DEPS];
    uint32_t dep_count;
    int (*run)(TemperGraph *graph, TemperAnalysis *analysis);
    bool (*is_enabled)(TemperOptLevel level);
} TemperPassImpl;
```

**Pass registry (in dependency order):**

| Pass | ID | Depends On | Level | Description |
|------|----|------------|-------|-------------|
| Dead Elimination | 0 | — | O1 | Remove ops whose output is never used |
| Constant Folding | 1 | — | O1 | Precompute constant expressions |
| Identity Elimination | 2 | — | O1 | Remove identity ops (add 0, mul 1, transpose^2) |
| Buffer Lifetime Planning | 3 | 0, 1, 2 | O2 | Analyze lifetimes, plan buffer reuse and aliasing |
| Operator Fusion | 4 | 0, 1, 2 | O2 | Fuse consecutive element-wise ops into one kernel |
| Kernel Batching | 5 | 4 | O3 | Group small ops into batch dispatches |
| Multi-Pass | 6 | 0-5 | O3 | Re-run O1 passes after fusion (may expose new dead nodes) |

### Topological Ordering

The pass manager runs passes in topological order based on dependencies:

```c
int temper_pass_manager_run(TemperPassManager *pm, TemperGraph *graph) {
    // 1. Compute topological order from dependencies
    uint32_t order[TEMPER_MAX_PASSES];
    temper_topological_sort(pm->passes, pm->pass_count, order);

    // 2. Run passes in order
    for (uint32_t i = 0; i < pm->pass_count; i++) {
        TemperPass *pass = &pm->passes[order[i]];

        if (!pass->enabled) continue;
        if (!pass->is_enabled(pm->opt_level)) continue;

        // 3. Run pass
        uint64_t start = temper_time_us();
        int ret = pass->run(graph, &analysis);
        uint64_t end = temper_time_us();

        // 4. Record stats
        pass->stats.runtime_ns = (end - start) * 1000;

        // 5. Invalidate analysis if pass changed the graph
        if (pass->stats.nodes_removed > 0 || pass->stats.ops_fused > 0) {
            temper_analysis_invalidate(&analysis);
        }
    }

    return 0;
}
```

### Buffer Lifetime Planning

This pass analyzes tensor lifetimes and plans memory reuse:

```c
int temper_buffer_lifetime_plan(TemperGraph *graph, TemperAnalysis *analysis) {
    // 1. Compute live ranges for every tensor
    //    live_start[node] = first use
    //    live_end[node] = last use

    // 2. Find non-overlapping tensors
    //    If tensor A dies before tensor B is born,
    //    they can share the same buffer.

    // 3. Create alias groups
    //    Group tensors that can share memory.

    // 4. Assign buffer slots
    //    Each alias group gets one buffer.
    //    Size = max(tensor sizes in group).

    // 5. Emit buffer reuse metadata
    //    Scheduler uses this for placement decisions.
}
```

Example:

```
Before:
  T1: live [0, 5]    — 4 bytes
  T2: live [3, 8]    — 4 bytes
  T3: live [6, 10]   — 4 bytes
  Peak: 12 bytes

After:
  T1 and T3 don't overlap → share buffer A
  T2 uses buffer B
  Peak: 8 bytes (saved 33%)
```

### Full pipeline API

```c
// Create pass manager
TemperPassManager *temper_pass_manager_create(TemperOptLevel level);

// Run full pipeline
int temper_compile(TemperGraph *graph, TemperOptLevel level);

// Individual phases (for testing/debugging)
int temper_verify_pre(TemperGraph *graph);
int temper_canonicalize(TemperGraph *graph);
TemperAnalysis temper_analyze(TemperGraph *graph);
int temper_optimize(TemperGraph *graph, TemperAnalysis *analysis, TemperOptLevel level);
int temper_verify_post(TemperGraph *graph);
```

## Consequences

### Enables
- Cleaner graphs for the scheduler (less work, better decisions)
- Memory reuse via buffer lifetime planning
- Kernel fusion reduces launch overhead
- Verification catches bugs early
- Pass statistics enable profiling and tuning
- Canonicalization simplifies all later passes

### Constrains
- Every pass must declare dependencies
- Passes must not break graph invariants
- Analysis must be invalidated when graph changes
- Verification runs twice (extra cost, but worth it)

### Tradeoffs
- **Compilation time vs. execution time:** Optimization takes time upfront but saves time during training. For long training runs, this is a net win.
- **Pass count vs. simplicity:** More passes = more optimization but more complexity. Topological ordering keeps it manageable.

## Alternatives Considered

### Option A: No compiler (raw graph execution)

**Pros:** Simple, no overhead.
**Cons:** Wasted work, no fusion, no reuse. Scheduler gets a messy graph.

### Option B: Single optimization pass

**Pros:** Simple.
**Cons:** Can't handle pass dependencies. Misses opportunities that require multiple passes.

### Option C: LLVM-style full compiler

**Pros:** Maximum optimization.
**Cons:** Massive complexity, overkill for v1. Our pipeline is simpler but sufficient.

**Chosen: Multi-pass with dependencies and analysis** — Balances optimization power with implementation complexity. Pass manager handles ordering. Analysis layer avoids redundant computation.

## Implementation Notes

### Pass dependency graph (directed acyclic)

```
Dead Elim ──────────┐
Constant Fold ──────┤
Identity Elim ──────┼──→ Buffer Lifetime ──→ Kernel Batching
                    │
Operator Fusion ────┘
                    │
                    └──→ Multi-Pass (re-runs O1 after fusion)
```

### Verification checks

```c
typedef enum TemperVerifyError {
    TEMPER_VERIFY_OK = 0,
    TEMPER_VERIFY_SHAPE_MISMATCH,     // tensor shapes incompatible
    TEMPER_VERIFY_DEVICE_MISMATCH,    // mixed devices without copy
    TEMPER_VERIFY_MISSING_GRADIENT,   // no path to compute gradient
    TEMPER_VERIFY_INVALID_RESOURCE,   // tensor points to freed resource
    TEMPER_VERIFY_CYCLE_DETECTED,     // graph has a cycle
    TEMPER_VERIFY_ORPHAN_NODE,        // node not connected to output
} TemperVerifyError;
```

### Analysis invalidation

```c
void temper_analysis_invalidate(TemperAnalysis *analysis) {
    // Mark all analysis as stale
    // Next pass that needs analysis will recompute
    analysis->valid = false;
}

TemperAnalysis *temper_analysis_get(TemperGraph *graph) {
    static TemperAnalysis cached = {0};
    if (!cached.valid || cached.graph_version != graph->version) {
        cached = temper_analyze(graph);
        cached.graph_version = graph->version;
        cached.valid = true;
    }
    return &cached;
}
```
