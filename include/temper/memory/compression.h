#ifndef TEMPER_COMPRESSION_H
#define TEMPER_COMPRESSION_H

#include "temper/core/resource.h"
#include <stdbool.h>
#include <stddef.h>

// Compression is a pluggable backend. The scheduler only knows:
//   can I compress this? → compress → opaque blob + compressed size
// All per-algorithm metadata (scale, zero-point, block tables, dictionaries,
// checksums) lives inside the opaque blob, owned by the backend.
typedef struct TemperCompressionBackend
{
    const char *name;
    bool (*can_compress)(const TemperResource *res);
    void *(*compress)(const float *src, size_t element_count, size_t *out_size);
    float *(*decompress)(const void *blob, size_t blob_size, size_t element_count);
    size_t (*estimate_size)(size_t element_count);
} TemperCompressionBackend;

// Phase 3 shipped backend: fp32 -> bf16 (truncation). Deterministic, ~2x density.
extern const TemperCompressionBackend temper_compression_bf16;

#endif
