# TemperEngine — Master Plan

**Version:** 0.1
**Language:** C17
**License:** MIT
**Mission:** Make AI training accessible on ordinary and older computers through a memory-first architecture.

---

## Vision

TemperEngine is not another AI framework. It is a rethink of how neural networks are trained, designed from scratch with simplicity, performance, portability, and memory efficiency as first-class goals.

**Core Innovation:** The Memory Scheduler — not an allocator, but a brain that manages tensor placement, eviction, recomputation, and compression across CPU, GPU, NPU, and SSD tiers.

**Ambition:** Train 1B parameter models on 4GB RAM.

---

## Core Principles

| Principle | Meaning |
|-----------|---------|
| Memory First | Every design decision starts with "how does this affect memory?" |
| CPU First, GPU Accelerated | CPU is always available. GPU is an optimization. |
| Modular Architecture | Every subsystem is independent and replaceable. |
| Zero Hidden Magic | Every operation is explicit and observable. |
| Research Friendly | Easy to add new ops, layers, optimizers, algorithms. |
| Cross Platform | macOS (Metal), Linux (CUDA), Windows (Vulkan). |
| Written in C | No C++, no Rust, no dependencies beyond system libraries. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User API                                 │
│            temper_add(a, b) · temper_matmul(a, b)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Graph Capture                               │
│         Records ops into tape · Builds computation graph        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Verification (Pre-Opt)                        │
│              Catches invalid user graphs early                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Graph Canonicalization                         │
│      Normalize equivalent forms (A+B → canonical order)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Layer                               │
│    Liveness · Dependencies · Aliases · Shapes · Devices         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADR-009: Optimization Pipeline                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Dead     │→ │ Constant │→ │ Identity │→ │ Operator │       │
│  │  Elim     │  │  Fold    │  │  Elim    │  │  Fusion  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                               │                                 │
│                      ┌────────▼────────┐                        │
│                      │  Buffer Lifetime │                       │
│                      │    Planning      │                       │
│                      └────────┬────────┘                        │
│                               │                                 │
│                      ┌────────▼────────┐                        │
│                      │ Kernel Batching  │                       │
│                      └─────────────────┘                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Verification (Post-Opt)                        │
│             Catches optimizer bugs before execution             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADR-004: Memory Scheduler                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Place   │→ │  Evict   │→ │Recompute │→ │ Compress │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADR-008: Runtime Dispatch                          │
│         API → Scheduler → Runtime (fast path bypass)            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┬─────────────...
               ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │   CPU    │   │  Metal   │   │  CUDA    │   ...
        │ Runtime  │   │ Runtime  │   │ Runtime  │
        └──────────┘   └──────────┘   └──────────┘
```

### Key design decisions

See ADR documents in `Planning/` directory:

| ADR | Decision |
|-----|----------|
| ADR-001 | TemperTensor handle + TemperResource indirection (hot/cold split, pooling, inline scalars) |
| ADR-002 | Device type + ID (not just "GPU") |
| ADR-003 | TemperRuntime function pointer table (capability cache, fast-path bypass) |
| ADR-004 | Multi-tier memory scheduler (per-device, sticky, async, batch, telemetry) |
| ADR-005 | Dynamic tape autodiff (arena alloc, node recycling, gradient pooling) |
| ADR-006 | Score-based recomputation policy |
| ADR-007 | Section-based `.temper` binary format |
| ADR-008 | Three-layer dispatch (command batching, lazy migration, small-op threshold) |
| ADR-009 | Graph optimization pipeline (pass manager, buffer lifetime planning, verification) |

---

## Memory Model

### Tiers

```
Tier 0: GPU VRAM       (fastest, smallest — ~8-24 GB)
Tier 1: CPU RAM        (medium speed, medium size — ~8-64 GB)
Tier 2: Compressed RAM (quantized, 2-4x density)
Tier 3: SSD            (slow, huge — ~256 GB-2 TB)
Tier 4: Recomputable   (deleted, can be regenerated from inputs)
```

### Scheduler logic

```
Tensor requested
  → Can this be recomputed cheaper than storing it?
  → YES: mark as recomputable, don't allocate
  → NO: where should it live?
    → Estimate FLOPs for ops that use it
    → Estimate copy cost between devices
    → Estimate launch overhead
    → Check memory pressure on each device
    → Choose placement
  → If target device is full:
    → What can be evicted?
    → What can be compressed?
    → What can be recomputed later?
    → Evict accordingly
```

### Recomputation policy

- Score = memory_saved / recompute_cost × (1 - access_frequency)
- Score > 1.0 → always recompute
- Score > 0.5 → recompute if pressured
- Score < 0.0 → store
- Pinned tensors → never evict

---

## File Structure

```
TemperEngine/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── PLAN.md
├── .clang-format
├── .editorconfig
├── .gitignore
├── Planning/                          (Architecture Decision Records)
│   ├── ADR-001-tensor-representation.md
│   ├── ADR-002-device-abstraction.md
│   ├── ADR-003-runtime-interface.md
│   ├── ADR-004-memory-scheduler.md
│   ├── ADR-005-autograd-graph.md
│   ├── ADR-006-recomputation-policy.md
│   ├── ADR-007-model-serialization.md
│   └── ADR-008-dispatch-model.md
├── include/
│   └── temper/
│       ├── temper.h
│       ├── core/
│       │   ├── memory.h
│       │   ├── threading.h
│       │   ├── logger.h
│       │   ├── profiler.h
│       │   ├── platform.h
│       │   ├── device.h              (NEW — ADR-002)
│       │   ├── resource.h            (NEW — ADR-001)
│       │   └── runtime.h             (RENAMED from backend.h — ADR-003)
│       ├── math/
│       │   ├── tensor.h              (MODIFIED — add resource ptr)
│       │   └── shape.h
│       ├── graph/
│       │   ├── tape.h                (MODIFIED — add checkpointing)
│       │   └── autodiff.h
│       ├── nn/
│       │   ├── layers.h
│       │   ├── activations.h
│       │   └── losses.h
│       ├── training/
│       │   ├── trainer.h             (MODIFIED — wire to scheduler)
│       │   ├── optimizer.h
│       │   ├── scheduler.h
│       │   └── checkpoint.h
│       ├── memory/
│       │   └── scheduler.h           (NEW — the brain)
│       ├── backend/
│       │   ├── cpu.h                 (NEW)
│       │   └── metal.h               (NEW — C declarations)
│       └── utils/
│           └── assert.h
├── src/
│   ├── temper.c
│   ├── core/
│   │   ├── memory.c
│   │   ├── threading.c
│   │   ├── logger.c
│   │   ├── profiler.c
│   │   └── platform.c
│   ├── math/
│   │   ├── tensor.c                 (MODIFIED — use TemperResource)
│   │   └── shape.c
│   ├── graph/
│   │   ├── tape.c                   (MODIFIED — real backward)
│   │   └── autodiff.c               (MODIFIED — chain rule)
│   ├── nn/
│   │   ├── layers.c                 (FIXED — add bias, real layernorm)
│   │   ├── activations.c
│   │   └── losses.c
│   ├── training/
│   │   ├── trainer.c                (MODIFIED — full training loop)
│   │   ├── optimizer.c
│   │   ├── checkpoint.c             (MODIFIED — save weights)
│   │   └── scheduler.c
│   ├── memory/
│   │   └── scheduler.c              (NEW — the brain)
│   └── backend/
│       ├── cpu.c                    (NEW)
│       ├── metal.m                  (NEW — Obj-C bridging)
│       ├── metal.h                  (NEW — C declarations)
│       └── metal_shaders.metal      (NEW — GPU kernels)
├── tests/
│   ├── CMakeLists.txt
│   ├── test_core.c
│   ├── test_math.c
│   ├── test_nn.c
│   ├── test_memory_scheduler.c      (NEW)
│   ├── test_metal.c                 (NEW)
│   └── test_dispatch.c              (NEW)
└── examples/
    ├── CMakeLists.txt
    ├── hello.c
    ├── train_mlp.c                  (NEW — MNIST proof of concept)
    └── train_transformer.c          (NEW — wikitext)
```

---

## Development Phases

### Phase 0 — Foundation ✅ (DONE)

- [x] Repository structure
- [x] CMake build system (C17, cross-platform)
- [x] Unit testing framework (33 tests passing)
- [x] Coding standards (.clang-format, .editorconfig)
- [x] Logging system
- [x] Assertions
- [x] Core runtime: memory (arena, pool), threading, platform detection
- [x] Tensor library: basic ops (add, sub, mul, div, matmul, transpose, reshape, sum)
- [x] NN layers: dense, embedding, layer norm, activations (ReLU, GELU, SiLU, softmax)
- [x] Training: SGD/Adam optimizers, cosine scheduler, checkpointing
- [x] MIT license

**Duration:** Completed
**Tests:** 33 passing

---

### Phase 0.5 — Architecture Decision Records (CURRENT)

- [x] ADR-001: Tensor Representation
- [x] ADR-002: Device Abstraction
- [x] ADR-003: Runtime Interface
- [x] ADR-004: Memory Scheduler
- [x] ADR-005: Autograd Graph
- [x] ADR-006: Recomputation Policy
- [x] ADR-007: Model Serialization
- [x] ADR-008: Dispatch Model
- [x] PLAN.md

**Duration:** Current session

---

### Phase 1 — Core Runtime Hardening ✅ (DONE)

**Goal:** Production-quality foundation.

**Tasks:**
- [x] Fix dense layer forward (add bias)
- [x] Implement layer norm forward properly (mean, variance, affine)
- [x] Add broadcasting to element-wise tensor ops
- [x] Add stride-aware tensor access
- [x] Thread pool unit tests
- [x] Memory pool stress tests
- [x] Cross-platform CI (GitHub Actions: macOS, Linux, Windows)

**Estimated duration:** 2 weeks  
**Actual Time Needed (Vibe Coding):** 2 days  
**Status:** Completed  
**Deliverable:** All existing code works correctly, broadcasting enabled, CI green.

---

### Phase 2 — Tensor Mobility (ADR-001, ADR-002) ✅ (DONE)

**Goal:** Tensors know where they live. Resources are tracked.

**Tasks:**
- [x] Implement `TemperResource` struct (ADR-001)
- [x] Implement `TemperDevice` type + ID (ADR-002)
- [x] Implement `TemperDeviceTable` for device registration
- [x] Implement `temper_tensor_to(device)` (creates new resource on target device)
- [x] Implement `temper_tensor_data()` with scheduler integration
- [x] Add resource field to `TemperTensor`
- [x] Resource tracking table in memory scheduler
- [x] CPU device as default
- [x] Migrate existing tensor ops to use TemperResource

**Estimated duration:** 1 week  
**Actual Time Needed (Vibe Coding):** 1 day  
**Status:** Completed  
**Deliverable:** `temper_tensor_to(tensor, GPU)` works. Resources are tracked via TemperResource indirection.

---

### Phase 3 — Memory Scheduler (ADR-004)

**Goal:** The brain of the engine.

**Tasks:**
- [ ] Implement `TemperMemScheduler` struct
- [ ] Tiered memory model (GPU, CPU, Compressed, SSD, Recomputable)
- [ ] Placement scoring function
- [ ] Eviction algorithm (score-based)
- [ ] Pressure tracking per tier
- [ ] Pinned tensor support
- [ ] Resource promotion/demotion
- [ ] Integration with tensor creation/access
- [ ] Memory scheduler unit tests
- [ ] Statistics tracking (evictions, recomputations, compressions)

**Estimated duration:** 2-3 weeks

**Deliverable:** Scheduler manages all tensor memory. Eviction works. Pressure tracking works.

---

### Phase 4 — Metal Backend (ADR-003) (macOS)

**Goal:** GPU compute via Metal.

**Tasks:**
- [ ] Obj-C bridging shim (`metal.h` / `metal.m`)
- [ ] Metal device + command queue creation
- [ ] Zero-copy buffer allocation (`mmap` + `newBufferWithBytesNoCopy`)
- [ ] Compute kernel compilation pipeline (`.metal` → `.metallib`)
- [ ] Element-wise GPU kernels:
  - [ ] add (element-wise)
  - [ ] sub (element-wise)
  - [ ] mul (element-wise)
  - [ ] div (element-wise)
  - [ ] relu
  - [ ] gelu
  - [ ] silu
- [ ] MPS matmul integration (via MPSMatrixMultiplication)
- [ ] `TemperRuntime` implementation for Metal (ADR-003)
- [ ] `temper_tensor_to(GPU)` with zero-copy
- [ ] Metal runtime unit tests
- [ ] GPU kernel tests

**Estimated duration:** 3-4 weeks

**Deliverable:** `temper_tensor_to(tensor, GPU)` → GPU compute → `temper_tensor_to(result, CPU)` works end to end.

---

### Phase 5 — Compute Scheduler (ADR-008)

**Goal:** Ops dispatch to the right device automatically.

**Tasks:**
- [ ] Implement dispatch model (API → Scheduler → Runtime)
- [ ] Device promotion logic (promote to GPU if any input is on GPU)
- [ ] Fallback chain (GPU → CPU → compressed)
- [ ] Small tensor threshold (< 4KB stays on CPU)
- [ ] Op registry per runtime (which ops each device supports)
- [ ] Integration with all tensor ops
- [ ] Transparent to user API
- [ ] Dispatch unit tests

**Estimated duration:** 1 week

**Deliverable:** `temper_add(a, b)` automatically dispatches to GPU if inputs are on GPU.

---

### Phase 6 — Autodiff + Training Loop (ADR-005)

**Goal:** Actually train models.

**Tasks:**
- [ ] Add `grad` field to `TemperResource`
- [ ] Implement backward for every op:
  - [ ] add, sub, mul, div
  - [ ] matmul (both inputs)
  - [ ] relu, gelu, silu
  - [ ] softmax
  - [ ] layer_norm
  - [ ] embedding
  - [ ] reshape, transpose
- [ ] Gradient accumulation (`temper_tensor_accumulate_grad`)
- [ ] Wire trainer: forward → loss → backward → optimizer_step → zero_grad
- [ ] Fix all NN layer backward passes
- [ ] Gradient clipping
- [ ] Train small MLP on MNIST (proof of concept)
- [ ] Autodiff unit tests

**Estimated duration:** 4-6 weeks

**Deliverable:** Train a 2-layer MLP on MNIST. Loss decreases. Accuracy > 90%.

---

### Phase 7 — Memory Scheduler Advanced (ADR-006)

**Goal:** Recomputation and compression.

**Tasks:**
- [ ] Recomputation scoring function
- [ ] Recomputation chain resolution (cascade)
- [ ] Gradient checkpointing integration with tape
- [ ] Automatic checkpoint interval selection
- [ ] Tensor compression (quantize evicted tensors)
- [ ] Compressed tier implementation
- [ ] SSD paging (swap cold tensors to disk)
- [ ] Pipeline parallelism (CPU loads batch N+1 while GPU trains batch N)
- [ ] Train transformer on wikitext (demonstrate memory efficiency)
- [ ] Memory efficiency benchmarks vs. PyTorch

**Estimated duration:** 4-6 weeks

**Deliverable:** Train a transformer on wikitext with 2x less memory than PyTorch.

---

### Phase 8 — Neural Network Library (Full)

**Goal:** Production-quality layers.

**Tasks:**
- [ ] Multi-Head Attention (self-attention, cross-attention)
- [ ] Transformer block (attention + FFN + residual + norm)
- [ ] Conv1d, Conv2d
- [ ] BatchNorm, RMSNorm
- [ ] Dropout (training + eval modes)
- [ ] LSTM, GRU
- [ ] Positional encoding (sinusoidal, learned)
- [ ] All backward passes
- [ ] Layer unit tests

**Estimated duration:** 4-6 weeks

**Deliverable:** Can define and train a full transformer model.

---

### Phase 9 — Dataset Pipeline

**Goal:** Efficient data loading.

**Tasks:**
- [ ] Dataset abstraction (`TemperDataset`)
- [ ] DataLoader with prefetching
- [ ] Memory-mapped datasets
- [ ] Streaming from SSD
- [ ] Multi-threaded loading (using existing thread pool)
- [ ] Data transforms (tokenization, augmentation)
- [ ] Integration with training loop
- [ ] DataLoader benchmarks

**Estimated duration:** 2-3 weeks

**Deliverable:** Load and batch data efficiently without blocking training.

---

### Phase 10 — Model Serialization (ADR-007)

**Goal:** Save and load models.

**Tasks:**
- [ ] `.temper` file format implementation
- [ ] Header + section table
- [ ] Weight serialization
- [ ] Optimizer state serialization
- [ ] Architecture serialization
- [ ] Metadata section (JSON)
- [ ] xxhash64 checksums
- [ ] Checkpoint/resume training
- [ ] Model export (inference only)
- [ ] Serialization unit tests

**Estimated duration:** 1-2 weeks

**Deliverable:** Save model, load model, resume training from checkpoint.

---

### Phase 11 — Graph Optimization Pipeline (ADR-009)

**Goal:** Compiler infrastructure for computation graphs.

**Tasks:**
- [ ] Pass Manager implementation
- [ ] Graph canonicalization
- [ ] Analysis layer (liveness, dependencies, aliases, shapes, devices)
- [ ] Dead tensor elimination pass
- [ ] Constant folding pass
- [ ] Identity elimination pass
- [ ] Operator fusion pass
- [ ] Buffer Lifetime Planning pass
- [ ] Kernel batching pass
- [ ] Verification pass (pre-opt and post-opt)
- [ ] Optimization levels (O0-O3)
- [ ] Pass statistics and profiling
- [ ] Graph compiler unit tests

**Estimated duration:** 3-4 weeks

**Deliverable:** Fused kernels run 2-3x faster. Buffer reuse reduces peak memory.

---

### Phase 12 — Performance Hardening

**Goal:** Optimize scheduler, runtime, and memory subsystems.

**Tasks:**
- [ ] Fast-path bypass for scheduler
- [ ] Per-device schedulers (no global lock)
- [ ] Hot/cold resource split
- [ ] Tensor pooling and slab allocation
- [ ] Inline scalars
- [ ] Runtime pointer caching
- [ ] Runtime capability cache
- [ ] Decision caching for placement
- [ ] Command batching
- [ ] Lazy migration
- [ ] Small-op threshold
- [ ] Telemetry collection
- [ ] Self-tuning scheduler
- [ ] Performance benchmarks

**Estimated duration:** 2-3 weeks

**Deliverable:** 90%+ fast-path bypass rate. Scheduler overhead negligible.

---

### Phase 13 — CUDA Backend

**Goal:** NVIDIA GPU support.

**Tasks:**
- [ ] CUDA runtime implementation
- [ ] cuBLAS integration for GEMM
- [ ] cuDNN integration for convolutions
- [ ] CUDA memory management
- [ ] Multi-GPU support (data parallel)
- [ ] CUDA kernel tests

**Estimated duration:** 4-6 weeks

**Deliverable:** Train on NVIDIA GPUs with CUDA acceleration.

---

### Phase 13 — Vulkan Backend

**Goal:** Cross-platform GPU support (Linux, Windows).

**Tasks:**
- [ ] Vulkan compute pipeline
- [ ] SPIR-V shader compilation
- [ ] Vulkan memory management
- [ ] Cross-platform GPU compute
- [ ] Vulkan kernel tests

**Estimated duration:** 4-6 weeks

**Deliverable:** GPU compute on Linux and Windows without CUDA.

---

### Phase 14 — CLI + Tooling

**Goal:** Developer experience.

**Tasks:**
- [ ] `temper init` — scaffold new project
- [ ] `temper train` — run training from config
- [ ] `temper test` — evaluate model
- [ ] `temper benchmark` — performance testing
- [ ] `temper export` — export model (ONNX, etc.)
- [ ] `temper profile` — memory/compute profiling
- [ ] `temper convert` — convert from PyTorch/ONNX

**Estimated duration:** 2-3 weeks

**Deliverable:** Command-line tools for common workflows.

---

### Phase 15 — Profiler

**Goal:** Visibility into what the engine is doing.

**Tasks:**
- [ ] RAM usage tracking (per tier)
- [ ] GPU memory tracking
- [ ] Tensor lifetime analysis (when allocated, when freed)
- [ ] Kernel timing (per-op profiling)
- [ ] Graph execution timing
- [ ] Training speed metrics (samples/sec, tokens/sec)
- [ ] Memory efficiency metrics (bytes per parameter, utilization)
- [ ] Visual profiler output (text-based, terminal-friendly)
- [ ] Export profiling data (JSON)

**Estimated duration:** 2 weeks

**Deliverable:** Full visibility into engine behavior. Can identify bottlenecks.

---

### Phase 16 — Plugin SDK

**Goal:** Extensibility without modifying the engine.

**Tasks:**
- [ ] Plugin API definition
- [ ] Custom layer registration
- [ ] Custom optimizer registration
- [ ] Custom loss function registration
- [ ] Custom dataset registration
- [ ] Custom runtime registration
- [ ] Plugin loading (dynamic library, `dlopen`/`LoadLibrary`)
- [ ] Plugin tests

**Estimated duration:** 2-3 weeks

**Deliverable:** Users can add new functionality without touching engine code.

---

## Timeline

| Phase | Duration | Cumulative | Milestone |
|-------|----------|------------|-----------|
| 0 — Foundation | Done | Week 0 | ✅ 33 tests passing |
| 0.5 — ADRs | 1 week | Week 1 | Architecture documented |
| 1 — Core Hardening | 2 weeks | Week 3 | Bug fixes, broadcasting |
| 2 — Tensor Mobility | 1 week | Week 4 | TemperResource, device tracking |
| 3 — Memory Scheduler | 3 weeks | Week 7 | The brain works |
| 4 — Metal Backend | 4 weeks | Week 11 | GPU compute on macOS |
| 5 — Compute Scheduler | 1 week | Week 12 | Transparent dispatch |
| 6 — Autodiff + Training | 6 weeks | Week 18 | **MVP: train MLP on MNIST** |
| 7 — Memory Scheduler v2 | 6 weeks | Week 24 | **Recomputation, checkpointing** |
| 8 — NN Library | 6 weeks | Week 30 | Full transformer support |
| 9 — Dataset Pipeline | 3 weeks | Week 33 | Efficient data loading |
| 10 — Serialization | 2 weeks | Week 35 | Save/load models |
| 11 — Graph Optimization Pipeline | 4 weeks | Week 39 | **Compiler: fusion, buffer planning** |
| 12 — Performance Hardening | 3 weeks | Week 42 | **Fast-path, batching, telemetry** |
| 13 — CUDA Backend | 6 weeks | Week 48 | NVIDIA GPU support |
| 14 — Vulkan Backend | 6 weeks | Week 54 | Cross-platform GPU |
| 15 — CLI + Tooling | 3 weeks | Week 57 | Developer experience |
| 16 — Profiler | 2 weeks | Week 59 | Full visibility |
| 17 — Plugin SDK | 3 weeks | Week 62 | Extensibility |

### Key milestones

| Week | Milestone |
|------|-----------|
| 1 | Architecture documented (ADRs complete) |
| 7 | Memory scheduler working |
| 11 | Metal GPU compute working |
| 18 | **MVP: train a model end-to-end** |
| 24 | Memory-efficient training with recomputation |
| 30 | Full transformer support |
| 39 | Graph optimization pipeline (fusion, buffer planning) |
| 42 | Performance hardening (fast-path, batching) |
| 48 | CUDA backend (NVIDIA GPUs) |
| 62 | Full vision complete |

---

## Success Criteria

TemperEngine succeeds when:

| Criterion | Metric |
|-----------|--------|
| Runs on old computers | Trains models on 4GB RAM laptop |
| Memory-efficient | 3-4x less memory than PyTorch for same model |
| Transparent | User writes `temper_add(a,b)`, engine handles everything |
| Cross-platform | macOS (Metal), Linux (CUDA), Windows (Vulkan) |
| Embeddable | C library, no runtime dependencies, < 1MB binary |
| Approachable | Clean API, good docs, ADRs explain every decision |
| Research-friendly | Easy to add new ops, layers, optimizers in < 100 lines |

---

## Testing Strategy

| Level | Framework | Coverage |
|-------|-----------|----------|
| Unit tests | Custom (assert-based) | All modules |
| Integration tests | CTest | Cross-module workflows |
| Memory tests | Custom | Scheduler, eviction, recomputation |
| GPU tests | Metal/CUDA specific | Backend correctness |
| Performance tests | Custom benchmarks | Throughput, latency, memory |
| Stress tests | Custom | Large models, low memory |

### CI/CD

```yaml
# GitHub Actions
matrix:
  os: [macos-latest, ubuntu-latest, windows-latest]
  compiler: [gcc, clang, msvc]
  build_type: [Debug, Release]

steps:
  - cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}
  - cmake --build build
  - ctest --test-dir build --output-on-failure
```

---

## Philosophy

TemperEngine is built on one belief:

> Intelligence should not require expensive hardware.

Our goal is to forge an AI engine that enables anyone — from students with decade-old laptops to researchers with high-end workstations — to build, train, and explore machine learning systems through a clean, efficient, and memory-first architecture.

---

## Appendix: Current State (Phase 0)

### What exists

| Module | Status | Quality |
|--------|--------|---------|
| Core (memory, threading, logger, profiler, platform) | Working | 7/10 |
| Math (tensor, shape) | Working (basic) | 3/10 |
| Graph (tape, autodiff) | Stubs | 1/10 |
| NN (layers, activations, losses) | Partial | 3/10 |
| Training (optimizer, trainer, scheduler) | Partial | 2/10 |

### What's missing (critical path)

1. Autodiff backward passes (nothing trains without this)
2. Memory scheduler (the core innovation)
3. Device abstraction (multi-device support)
4. Metal backend (GPU acceleration)
5. Training loop wiring (forward → loss → backward → step)

### What's working

- Arena and pool allocators
- Thread pool with job scheduling
- Tensor creation, destruction, basic ops
- Element-wise add, sub, mul, div
- Matrix multiply (2D, naive)
- Transpose, reshape, sum
- ReLU, GELU, SiLU, Softmax
- Cross-entropy, MSE losses
- SGD, Adam optimizers (correct math)
- Cosine annealing scheduler
- Checkpoint save/load (basic)
- All 33 unit tests passing

---

*This document is the source of truth for TemperEngine's architecture and development roadmap.*
