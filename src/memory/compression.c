#include "temper/memory/compression.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Blob layout: [uint64 element_count][uint16 bf16 data ...]
// Fully backend-owned; the scheduler treats it as opaque.

static bool bf16_can_compress(const TemperResource *res)
{
    if (!res)
    {
        return false;
    }
    if (!(res->flags & TEMPER_RESOURCE_RESIDENT))
    {
        return false;
    }
    if (res->flags & TEMPER_RESOURCE_COMPRESSED)
    {
        return false;
    }
    if (res->host_ptr == NULL)
    {
        return false;
    }
    return res->bytes >= 64; // skip tiny tensors
}

static void *bf16_compress(const float *src, size_t element_count, size_t *out_size)
{
    if (!src || !out_size || element_count == 0)
    {
        return NULL;
    }
    size_t blob_size = sizeof(uint64_t) + element_count * sizeof(uint16_t);
    uint8_t *blob = (uint8_t *)malloc(blob_size);
    if (!blob)
    {
        return NULL;
    }

    memcpy(blob, &element_count, sizeof(uint64_t));
    uint16_t *dst = (uint16_t *)(blob + sizeof(uint64_t));
    for (size_t i = 0; i < element_count; i++)
    {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(uint32_t));
        dst[i] = (uint16_t)(bits >> 16); // truncate fp32 -> bf16
    }
    *out_size = blob_size;
    return blob;
}

static float *bf16_decompress(const void *blob, size_t blob_size, size_t element_count)
{
    if (!blob)
    {
        return NULL;
    }
    if (blob_size < sizeof(uint64_t))
    {
        return NULL;
    }
    uint64_t stored;
    memcpy(&stored, blob, sizeof(uint64_t));
    if (stored != element_count)
    {
        return NULL;
    }
    if (blob_size != sizeof(uint64_t) + element_count * sizeof(uint16_t))
    {
        return NULL;
    }

    const uint16_t *src = (const uint16_t *)((const uint8_t *)blob + sizeof(uint64_t));
    float *dst = (float *)malloc(element_count * sizeof(float));
    if (!dst)
    {
        return NULL;
    }
    for (size_t i = 0; i < element_count; i++)
    {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &bits, sizeof(uint32_t));
    }
    return dst;
}

static size_t bf16_estimate_size(size_t element_count)
{
    return sizeof(uint64_t) + element_count * sizeof(uint16_t);
}

const TemperCompressionBackend temper_compression_bf16 = {
    "bf16",
    bf16_can_compress,
    bf16_compress,
    bf16_decompress,
    bf16_estimate_size
};
