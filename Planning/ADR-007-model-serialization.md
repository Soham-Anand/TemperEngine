# ADR-007: Model Serialization

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** Soham Anand

## Context

We need to save and load:
- Model weights
- Architecture definition
- Optimizer state (moments, step count)
- Training state (epoch, step, loss)
- Metadata (version, description, tags)

The format must be:
- Efficient to read/write (no parsing overhead for large models)
- Versioned (backward compatibility across TemperEngine versions)
- Checksummed (data integrity verification)
- Section-based (partial loading — load weights without optimizer state)

## Decision

`.temper` binary format with section-based layout.

### File structure

```
┌──────────────────────────────────────────┐
│ Header (64 bytes)                        │
│   magic: "TEMPER" (6 bytes)              │
│   version: u32                           │
│   section_count: u32                     │
│   flags: u32                             │
│   timestamp: u64                         │
│   checksum: u64 (xxhash of all sections) │
│   reserved: 24 bytes                     │
├──────────────────────────────────────────┤
│ Section Table (32 bytes × N)             │
│   type: u32                              │
│   offset: u64                            │
│   size: u64                              │
│   checksum: u64                          │
│   flags: u32                             │
├──────────────────────────────────────────┤
│ Section: Architecture                    │
│   layer_count: u32                       │
│   For each layer:                        │
│     type: u32 (DENSE, LAYERNORM, etc.)   │
│     params: variable                     │
├──────────────────────────────────────────┤
│ Section: Weights                         │
│   param_count: u32                       │
│   For each parameter:                    │
│     name_len: u32                        │
│     name: bytes                          │
│     ndim: u8                             │
│     dims: i64[ndim]                      │
│     dtype: u32                           │
│     data: raw bytes                      │
├──────────────────────────────────────────┤
│ Section: Optimizer State                 │
│   optimizer_type: u32 (SGD, ADAM)        │
│   param_count: u32                       │
│   For each parameter:                    │
│     m_len: u64                           │
│     m: raw bytes (first moment)          │
│     v_len: u64                           │
│     v: raw bytes (second moment)         │
│   step: u32                              │
├──────────────────────────────────────────┤
│ Section: Training State                  │
│   epoch: u32                             │
│   step: u32                              │
│   loss: f32                              │
│   best_loss: f32                         │
│   rng_state_len: u64                     │
│   rng_state: bytes                       │
├──────────────────────────────────────────┤
│ Section: Metadata (JSON)                 │
│   len: u32                               │
│   json: bytes                            │
│   {                                      │
│     "description": "...",                │
│     "tags": ["transformer", "nlp"],      │
│     "author": "...",                     │
│     "temper_version": "0.1.0"            │
│   }                                      │
└──────────────────────────────────────────┘
```

### Section types

```c
typedef enum TemperSectionType {
    TEMPER_SECTION_HEADER      = 0,
    TEMPER_SECTION_ARCHITECTURE = 1,
    TEMPER_SECTION_WEIGHTS     = 2,
    TEMPER_SECTION_OPTIMIZER   = 3,
    TEMPER_SECTION_TRAINING    = 4,
    TEMPER_SECTION_METADATA    = 5,
} TemperSectionType;
```

### Section table entry

```c
typedef struct TemperSectionEntry {
    uint32_t type;
    uint32_t flags;         // compressed, encrypted, etc.
    uint64_t offset;        // byte offset from start of file
    uint64_t size;          // section size in bytes
    uint64_t checksum;      // xxhash64 of section data
} TemperSectionEntry;
```

### API

```c
// Full save/load
int temper_save(const char *path, const TemperModel *model);
int temper_load(const char *path, TemperModel *model);

// Partial loading
int temper_load_weights_only(const char *path, TemperModel *model);
int temper_load_metadata(const char *path, TemperMetadata *meta);
int temper_load_optimizer(const char *path, TemperOptimizer *opt);

// Inspection
int temper_file_info(const char *path, TemperFileInfo *info);
bool temper_file_verify(const char *path);
```

### Checksum verification

```c
bool temper_file_verify(const char *path) {
    FILE *f = fopen(path, "rb");
    TemperHeader header;
    fread(&header, sizeof(header), 1, f);

    // Verify header checksum
    uint64_t file_checksum = header.checksum;
    header.checksum = 0;  // zero out for calculation
    uint64_t calculated = xxhash64(&header, sizeof(header));

    // Verify section checksums
    for (uint32_t i = 0; i < header.section_count; i++) {
        TemperSectionEntry entry;
        fread(&entry, sizeof(entry), 1, f);

        fseek(f, entry.offset, SEEK_SET);
        void *data = malloc(entry.size);
        fread(data, entry.size, 1, f);

        uint64_t section_checksum = xxhash64(data, entry.size);
        if (section_checksum != entry.checksum) {
            free(data);
            fclose(f);
            return false;
        }
        free(data);
        fseek(f, sizeof(TemperHeader) + (i + 1) * sizeof(TemperSectionEntry), SEEK_SET);
    }

    fclose(f);
    return true;
}
```

### Compression support

Sections can be optionally compressed:

```c
typedef struct TemperSectionEntry {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;           // compressed size
    uint64_t uncompressed_size;  // original size
    uint64_t checksum;
} TemperSectionEntry;
```

Compression is applied per-section. Weights section benefits most (2-3x smaller files).

## Consequences

### Enables
- Checkpoint/resume training
- Model sharing between users
- Migration between hardware (CPU ↔ GPU)
- Partial loading (load weights without optimizer for inference)
- Data integrity verification

### Constrains
- Binary format requires versioning discipline
- Backward compatibility must be maintained across versions
- Format is not human-readable (JSON metadata section helps)

### Tradeoffs
- **Binary vs. text:** Binary is fast and compact. JSON metadata provides human readability where it matters.
- **Section-based vs. flat:** Sections enable partial loading and forward compatibility (unknown sections can be skipped).

## Alternatives Considered

### Option A: JSON (like ONNX)

**Pros:** Human-readable, standard tools can inspect.
**Cons:** Slow for large models, no raw tensor storage without encoding.

### Option B: Pickle (like PyTorch)

**Pros:** Simple, supports arbitrary Python objects.
**Cons:** Security risks (arbitrary code execution), Python dependency.

### Option C: FlatBuffers (Google)

**Pros:** Zero-copy deserialization, schema evolution.
**Cons:** External dependency, adds build complexity.

### Option D: Custom text format

**Pros:** Human-readable.
**Cons:** Parsing overhead, fragile, large file sizes.

**Chosen: Section-based binary** — Compact, fast, versioned, supports partial loading. JSON metadata provides human-readable info.

## Implementation Notes

### xxhash (lightweight, fast)

We use xxhash64 for checksums. It's fast (40 GB/s on modern hardware) and has excellent distribution. We can implement it in ~50 lines of C or use the single-file implementation.

### Endianness

All multi-byte values are stored in little-endian format. On big-endian systems, we swap during read/write. (Most modern systems are little-endian.)

### Version migration

```c
// When loading, check version compatibility
if (header.version > TEMPER_FORMAT_VERSION) {
    temper_error("File format version %u is newer than supported %u",
                 header.version, TEMPER_FORMAT_VERSION);
    return -1;
}

// Skip unknown sections (forward compatibility)
for (uint32_t i = 0; i < header.section_count; i++) {
    if (!temper_section_known(entry.type)) {
        fseek(f, entry.size, SEEK_CUR);
        continue;
    }
    // ... load section
}
```
