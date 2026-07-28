# ADR-006: Recomputation Policy

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

When memory is tight, we have two choices for a tensor we no longer need immediately:
1. **Store it** — keep it in memory (costs bytes)
2. **Recompute it** — free it, regenerate later from inputs (costs FLOPs)

The right choice depends on:
- How expensive the tensor was to compute (FLOPs)
- How much memory it occupies (bytes)
- Whether its inputs are still available
- How often it will be needed again

This is the key insight that allows TemperEngine to train models larger than available memory.

## Decision

Score-based recomputation policy with cascading support.

### Recomputation score

```c
float temper_recompute_score(TemperResource *r) {
    // Score > 0 means recompute wins (evict and regenerate)
    // Score < 0 means store wins (keep in memory)

    if (!r->recomputable || !r->origin) {
        return -1.0f;  // cannot recompute, must store
    }

    float memory_saved = (float)r->bytes;
    float recompute_cost = (float)r->origin->recompute_flops;

    // Avoid division by zero
    if (recompute_cost < 1.0f) recompute_cost = 1.0f;

    // Access frequency: how often is this tensor needed?
    float access_rate = (float)r->access_count / (float)(r->lifetime + 1);

    // Score = memory efficiency / access penalty
    // High score = good candidate for recomputation
    float score = (memory_saved / recompute_cost) * (1.0f - access_rate);

    return score;
}
```

### Decision matrix

| Score | Action | Tier |
|-------|--------|------|
| > 1.0 | Always recompute | Tier 4 (deleted) |
| > 0.5 | Recompute if pressured | Tier 2-3 (compressed/SSD) |
| > 0.0 | Recompute as last resort | Tier 3 (SSD) |
| < 0.0 | Store | Tier 0-1 (GPU/CPU) |
| pinned | Never evict | Current tier |

### Recomputation chain

When we recompute a tensor, we may need its inputs. If those inputs were also evicted:

```
Need tensor T
  → T was recomputable, freed
  → Need inputs I1, I2
    → I1 is still live → use it
    → I2 was also freed → recompute I2 from its inputs
      → I2's inputs are live → recompute I2 → recompute T
```

The scheduler tracks the full recomputation graph and can cascade:

```c
int temper_recompute(TemperResource *r) {
    if (!r->recomputable || !r->origin) {
        return -1;  // cannot recompute
    }

    TemperGraphNode *origin = r->origin;

    // 1. Check if all inputs are available
    for (uint32_t i = 0; i < origin->input_count; i++) {
        TemperResource *input_res = origin->inputs[i]->resource;
        if (input_res->tier == TIER_RECOMPUTABLE) {
            // Cascade: recompute this input first
            int ret = temper_recompute(input_res);
            if (ret != 0) return ret;  // failed to recompute
        }
    }

    // 2. Re-execute the op
    TemperRuntime *rt = temper_get_runtime(origin->output.resource->device);
    rt->dispatch(origin->op, origin->inputs, origin->input_count, &origin->output);

    // 3. Update resource state
    r->tier = origin->output.resource->device == TEMPER_DEVICE_GPU ? TIER_GPU : TIER_CPU;
    r->last_access = temper_time_us();

    return 0;
}
```

### When NOT to recompute

| Condition | Reason |
|-----------|--------|
| Op is expensive (large matmul, attention) | Recompute cost exceeds memory savings |
| Inputs are also gone (cascade too deep) | Cascading recomputation is expensive |
| Tensor is accessed frequently (hot tensor) | Would recompute constantly |
| Tensor is needed every backward pass | Activation in a transformer layer |
| Recomputation FLOPs > 10x memory saved | Poor return on investment |

### When TO recompute

| Condition | Reason |
|-----------|--------|
| Op is cheap (add, relu, reshape) | Recomputation is nearly free |
| Inputs are still live (no cascade needed) | Simple regeneration |
| Tensor is accessed rarely | Intermediate in a skip connection |
| Tensor is large but cheap to regenerate | High memory savings, low compute cost |
| Training is memory-bound, not compute-bound | Have spare FLOPs |

### Gradient checkpointing integration

For transformers, the natural checkpoint boundaries are layer boundaries:

```
Layer 0: forward → checkpoint
Layer 1: forward → free (recompute later)
Layer 2: forward → free (recompute later)
Layer 3: forward → checkpoint
...
Layer N: forward → keep (loss)
```

During backward:
- At checkpoint 3: recompute layers 1-3 from checkpoint 0
- At checkpoint N: recompute layers 4-N from checkpoint 3

This gives O(sqrt(N)) memory for N layers.

### Cost model

```c
typedef struct TemperRecomputeCost {
    size_t bytes_saved;        // memory freed by evicting
    size_t recompute_flops;    // FLOPs to regenerate
    uint32_t input_count;      // how many inputs needed
    uint32_t inputs_available; // how many are still live
    bool cascade_risk;         // could this trigger cascading recomputation?
} TemperRecomputeCost;
```

The scheduler uses this to make informed decisions.

## Consequences

### Enables
- Training models 3-4x larger than memory capacity
- Automatic memory management without user intervention
- Gradient checkpointing with minimal user effort
- Adaptive behavior: fast on fast hardware, memory-efficient on constrained hardware

### Constrains
- Need accurate FLOP counting for each op (must be implemented per-op)
- Recomputation adds compute overhead (trade memory for time)
- Cascade recomputation can be expensive if not bounded

### Tradeoffs
- **Memory vs. compute:** Users can tune the threshold to trade speed for capacity.
- **Simplicity vs. accuracy:** Simple score is good enough. Complex model (ML-based prediction) is overkill.

## Alternatives Considered

### Option A: Always recompute

**Pros:** Maximum memory savings.
**Cons:** Too slow for expensive ops. Wastes compute unnecessarily.

### Option B: Never recompute

**Pros:** Maximum speed.
**Cons:** Wastes memory on cheap ops. Defeats the purpose.

### Option C: Fixed checkpoint interval

**Pros:** Predictable.
**Cons:** Doesn't adapt to op costs or memory pressure.

### Option D: Manual checkpointing (PyTorch `torch.utils.checkpoint`)

**Pros:** User control.
**Cons:** Users shouldn't have to decide. Automatic is better for research.

**Chosen: Score-based with cascade support** — The scheduler makes intelligent decisions based on actual costs, not heuristics.

## Implementation Notes

### FLOP counting per op

```c
size_t temper_op_flops(TemperOpType op, const TemperTensor *const *inputs, uint32_t count) {
    switch (op) {
    case TEMPER_OP_ADD:
    case TEMPER_OP_SUB:
    case TEMPER_OP_MUL:
    case TEMPER_OP_DIV:
        return temper_shape_count(&inputs[0]->shape);  // element-wise
    case TEMPER_OP_MATMUL: {
        // M*N*K multiply
        int64_t M = inputs[0]->shape.dims[0];
        int64_t K = inputs[0]->shape.dims[1];
        int64_t N = inputs[1]->shape.dims[1];
        return 2 * M * N * K;  // multiply + add
    }
    case TEMPER_OP_RELU:
    case TEMPER_OP_GELU:
        return temper_shape_count(&inputs[0]->shape);
    default:
        return 0;
    }
}
```

### Configuration

```c
// Tune recomputation aggressiveness
void temper_scheduler_set_recompute_threshold(float threshold);  // default: 0.5

// Manual pinning (prevent eviction)
void temper_tensor_pin(TemperTensor *t);
void temper_tensor_unpin(TemperTensor *t);
```
