# ADR-005: Autograd Graph

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

Automatic differentiation is what makes training possible. Without it, TemperEngine is just a tensor library.

We need a dynamic computational graph that:
- Records operations during forward pass
- Supports reverse-mode autodiff (backpropagation)
- Integrates with the memory scheduler for gradient checkpointing
- Handles gradient accumulation
- Works with the recomputation policy

## Decision

Dynamic tape-based autodiff with memory scheduler integration.

### Tape structure

```c
typedef struct TemperTape {
    TemperGraphNode **nodes;      // ordered list of ops
    uint32_t node_count;
    uint32_t capacity;

    // Gradient checkpointing
    uint32_t checkpoint_interval; // every N nodes, pin a checkpoint
    uint32_t checkpoint_count;
    uint32_t next_checkpoint;     // index of next checkpoint node

    // Metadata
    uint64_t total_flops;         // total forward pass FLOPs
    size_t total_bytes;           // total intermediate bytes
} TemperTape;
```

### Graph node

```c
typedef struct TemperGraphNode {
    uint32_t id;
    TemperOpType op;

    // Inputs (up to 4 for simplicity, extendable)
    TemperTensor *inputs[4];
    uint32_t input_count;

    // Output
    TemperTensor output;

    // Gradient
    TemperTensor grad;           // gradient of loss w.r.t. this node's output

    // Backward function pointer
    void (*backward)(struct TemperGraphNode *node);

    // Recomputation support
    bool recomputable;           // can this op be replayed?
    size_t recompute_flops;      // cost to recompute (for scheduler)

    // Memory tracking
    bool is_checkpoint;          // is this a checkpoint node?
    bool grad_computed;          // has backward been called?
} TemperGraphNode;
```

### Backward pass

```c
void temper_tape_backward(TemperTape *tape) {
    // 1. Start from loss (last node)
    TemperGraphNode *loss = tape->nodes[tape->node_count - 1];
    loss->grad = temper_tensor_ones_like(&loss->output);

    // 2. Traverse in reverse
    for (int i = (int)tape->node_count - 1; i >= 0; i--) {
        TemperGraphNode *node = tape->nodes[i];

        // 3. Check if inputs need recomputation
        for (uint32_t j = 0; j < node->input_count; j++) {
            TemperResource *res = node->inputs[j]->resource;
            if (res->tier == TIER_RECOMPUTABLE) {
                // Recompute this input from its origin op
                temper_recompute(res);
            }
        }

        // 4. Compute gradients
        if (node->backward && !node->grad_computed) {
            node->backward(node);
            node->grad_computed = true;
        }

        // 5. Accumulate gradients into input nodes' grad fields
        //    (chain rule: dL/dx = dL/dy * dy/dx)

        // 6. Memory scheduler: can we free this node's output now?
        if (!node->is_checkpoint && !node->recomputable) {
            if (temper_scheduler_can_free(node->output.resource)) {
                temper_scheduler_queue_eviction(node->output.resource);
            }
        }
    }
}
```

### Gradient checkpointing

The tape works with the memory scheduler to decide which intermediates to keep:

```
Forward pass (with checkpointing every 3 nodes):
  Node 0 → keep (input)
  Node 1 → free (recomputable)
  Node 2 → free (recomputable)
  Node 3 → PIN (checkpoint) ← scheduler pins this
  Node 4 → free (recomputable)
  Node 5 → free (recomputable)
  Node 6 → PIN (checkpoint) ← scheduler pins this
  Node 7 → free (recomputable)
  Node 8 → keep (loss)

Backward pass:
  Node 8 → compute grad (loss is pinned)
  Node 7-3 → recompute nodes 7-3 from checkpoint 6, compute grads
  Node 2-0 → recompute nodes 2-0 from checkpoint 3, compute grads
```

This is O(sqrt(N)) memory instead of O(N) for a transformer with N layers.

### Automatic checkpointing

```c
void temper_tape_set_checkpoint_interval(TemperTape *tape, uint32_t interval) {
    tape->checkpoint_interval = interval;

    // Mark checkpoint nodes
    for (uint32_t i = 0; i < tape->node_count; i++) {
        if (i % interval == 0 || i == tape->node_count - 1) {
            tape->nodes[i]->is_checkpoint = true;
            // Pin checkpoint resources
            temper_scheduler_pin(tape->nodes[i]->output.resource);
        }
    }
}
```

### Op backward implementations

Each op needs a backward function:

```c
void temper_add_backward(TemperGraphNode *node) {
    // d(a+b)/da = 1, d(a+b)/db = 1
    // So grad passes through unchanged to both inputs
    temper_tensor_accumulate_grad(node->inputs[0]->grad, &node->grad);
    temper_tensor_accumulate_grad(node->inputs[1]->grad, &node->grad);
}

void temper_matmul_backward(TemperGraphNode *node) {
    // d(AB)/dA = grad * B^T
    // d(AB)/dB = A^T * grad
    TemperTensor grad_a = temper_tensor_matmul(&node->grad,
                                &temper_tensor_transpose(node->inputs[1]));
    TemperTensor grad_b = temper_tensor_matmul(
                                &temper_tensor_transpose(node->inputs[0]),
                                &node->grad);
    temper_tensor_accumulate_grad(node->inputs[0]->grad, &grad_a);
    temper_tensor_accumulate_grad(node->inputs[1]->grad, &grad_b);
}
```

## Consequences

### Enables
- Backpropagation through any computation
- Gradient checkpointing (O(sqrt(N)) memory)
- Gradient accumulation (for large batch simulation)
- Memory-efficient training of large models

### Constrains
- Forward pass must record ops (tape overhead: ~100 bytes per node)
- Each op needs a backward implementation
- Dynamic graph can't be optimized as aggressively as static

### Tradeoffs
- **Tape overhead vs. training capability:** 100 bytes per node is negligible vs. megabytes of tensor data.
- **Dynamic vs. static:** Dynamic is flexible for research. Static enables more optimization. Dynamic wins for our research-friendly goal.

## Alternatives Considered

### Option A: Static graph (TensorFlow 1.x)

**Pros:** Can optimize graph before execution. Efficient for inference.
**Cons:** Inflexible, bad for research, can't handle dynamic control flow.

### Option B: Source-level transformation (JAX)

**Pros:** Powerful composable transformations.
**Cons:** Complex, requires tracing or compilation, steep learning curve.

### Option C: Dual numbering (Autograd)

**Pros:** Efficient gradient computation.
**Cons:** Adds complexity beyond our needs.

**Chosen: Dynamic tape with checkpointing** — Simple, flexible, research-friendly. Checkpointing integrates naturally with the memory scheduler.

## Implementation Notes

### Tape creation

```c
TemperTape *temper_tape_create(void);
void temper_tape_destroy(TemperTape *tape);
```

### Recording ops

```c
TemperGraphNode *temper_tape_record(TemperTape *tape, TemperOpType op,
                                    TemperTensor *inputs, uint32_t input_count,
                                    TemperTensor output);
```

### Gradient accumulation

```c
void temper_tensor_accumulate_grad(TemperTensor *grad, const TemperTensor *delta);
```

This adds `delta` to `grad` (not replaces). This is needed for nodes with multiple consumers.

## Performance Mitigations

### Arena Allocator for Graph Nodes

Allocate all graph nodes from an arena. Reset per training step:

```c
typedef struct TemperGraphArena {
    char *buffer;
    size_t capacity;
    size_t offset;
} TemperGraphArena;

void *temper_graph_arena_alloc(TemperGraphArena *arena, size_t size) {
    size_t aligned = (size + 7) & ~(size_t)7;
    if (arena->offset + aligned > arena->capacity) return NULL;
    void *ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

void temper_graph_arena_reset(TemperGraphArena *arena) {
    arena->offset = 0;  // bulk free, no individual free calls
}
```

Per training step:
1. Reset arena (O(1) — just set offset to 0)
2. Forward pass allocates nodes from arena
3. Backward pass uses those nodes
4. Step complete — reset arena

No malloc/free overhead during the hot path.

### Graph Node Recycling

Instead of allocating new nodes, reuse freed nodes from a free list:

```c
typedef struct TemperNodePool {
    TemperGraphNode *free_list;
    uint32_t free_count;
    TemperGraphArena arena;  // backing arena for initial allocation
} TemperNodePool;

TemperGraphNode *temper_node_pool_acquire(TemperNodePool *pool) {
    if (pool->free_list) {
        TemperGraphNode *node = pool->free_list;
        pool->free_list = (TemperGraphNode *)node->next_free;
        pool->free_count--;
        return node;
    }
    // Free list empty — allocate from arena
    return temper_graph_arena_alloc(&pool->arena, sizeof(TemperGraphNode));
}

void temper_node_pool_release(TemperNodePool *pool, TemperGraphNode *node) {
    node->next_free = pool->free_list;
    pool->free_list = node;
    pool->free_count++;
}
```

### Gradient Tensor Pooling

Reuse gradient tensors of the same shape:

```c
typedef struct TemperGradPool {
    struct {
        TemperShape shape;
        TemperTensor *tensors;
        uint32_t count;
        uint32_t capacity;
    } buckets[64];  // 64 shape buckets
} TemperGradPool;

TemperTensor *temper_grad_pool_acquire(TemperGradPool *pool, TemperShape shape) {
    uint32_t bucket = temper_shape_hash(shape) % 64;
    TemperGradPoolBucket *b = &pool->buckets[bucket];

    for (uint32_t i = 0; i < b->count; i++) {
        if (temper_shape_equal(&b->tensors[i].shape, &shape)) {
            // Found reusable gradient tensor
            TemperTensor *t = &b->tensors[i];
            temper_tensor_zero(t);  // zero out data
            return t;
        }
    }

    // No reusable tensor — allocate new
    TemperTensor t = temper_tensor_create(shape, TEMPER_DTYPE_F32);
    if (b->count < b->capacity) {
        b->tensors[b->count++] = t;
    }
    return &b->tensors[b->count - 1];
}
```

### Tape Memory Budget

Limit tape size to prevent unbounded growth:

```c
#define TEMPER_MAX_TAPE_NODES (1 << 20)  // 1M nodes max

int temper_tape_record(TemperTape *tape, ...) {
    if (tape->node_count >= TEMPER_MAX_TAPE_NODES) {
        temper_error("Tape overflow: %u nodes", tape->node_count);
        return -1;
    }
    // ... record node
}
```

### Incremental Gradient Cleanup

Free gradients as soon as they're no longer needed:

```c
void temper_tape_backward_cleanup(TemperTape *tape, TemperGraphNode *node) {
    // After propagating gradient to inputs, free this node's gradient
    if (node->grad_computed && !node->is_checkpoint) {
        temper_tensor_destroy(&node->grad);
    }
}
```
