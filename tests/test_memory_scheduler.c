#include "temper/temper.h"
#include "temper/core/device.h"
#include "temper/core/resource.h"
#include "temper/memory/scheduler.h"
#include "temper/memory/compression.h"
#include "temper/math/tensor.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;
static int test_failed = 0;

#define TEST(name) static void name(void)
#define RUN(name)                                                             \
    do                                                                        \
    {                                                                         \
        tests_run++;                                                          \
        test_failed = 0;                                                      \
        printf("  %-40s", #name);                                             \
        name();                                                               \
        if (test_failed)                                                      \
            printf("FAIL\n");                                                 \
        else                                                                  \
        {                                                                     \
            tests_passed++;                                                   \
            printf("PASS\n");                                                 \
        }                                                                     \
    } while (0)

#define ASSERT(cond)                                                          \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            printf("\n    %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            test_failed = 1;                                                  \
            return;                                                           \
        }                                                                     \
    } while (0)

#define ASSERT_FLOAT(a, b, eps) (fabsf((a) - (b)) < (eps))

// Create a f32 tensor of `bytes` bytes (must be a multiple of 4) with
// 0.5-increment values starting at `base` (all exactly bf16-representable).
static TemperTensor make(size_t bytes, float base)
{
    size_t count = bytes / sizeof(float);
    if (count == 0)
    {
        count = 1;
    }
    TemperShape s = temper_shape_1d((int64_t)count);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *data = temper_tensor_data(&t);
    for (size_t i = 0; i < count; i++)
    {
        data[i] = base + (float)i * 0.5f;
    }
    return t;
}

TEST(test_scheduler_init_shutdown)
{
    temper_scheduler_init();
    ASSERT(temper_scheduler_initialized());
    ASSERT(temper_scheduler_version() == TEMPER_SCHEDULER_VERSION);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_CPU) > 0);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_GPU) > 0);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_COMPRESSED) > 0);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_SSD) > 0);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
    ASSERT(!temper_scheduler_initialized());
    ASSERT(temper_scheduler_validate()); // nothing to validate
}

TEST(test_tier_budget_override)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 1024);
    temper_scheduler_set_tier_budget(TEMPER_TIER_COMPRESSED, 2048);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_CPU) == 1024);
    ASSERT(temper_scheduler_tier_budget(TEMPER_TIER_COMPRESSED) == 2048);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 0);
    ASSERT(temper_scheduler_tier_logical(TEMPER_TIER_CPU) == 0);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_pressure_math)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 1024);
    ASSERT(temper_scheduler_pressure(TEMPER_TIER_CPU) < 0.01f);
    ASSERT(!temper_scheduler_under_pressure(TEMPER_TIER_CPU));

    TemperTensor t = make(512, 1.0f);
    ASSERT(temper_scheduler_pressure(TEMPER_TIER_CPU) > 0.4f);
    ASSERT(temper_scheduler_pressure(TEMPER_TIER_CPU) < 0.6f);

    temper_tensor_destroy(&t);
    ASSERT(temper_scheduler_pressure(TEMPER_TIER_CPU) < 0.01f);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_resource_register_tracking)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 8192);
    {
        TemperTensor a = make(256, 1.0f);
        TemperTensor b = make(512, 2.0f);
        ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 256 + 512);
        ASSERT(temper_scheduler_tier_logical(TEMPER_TIER_CPU) == 256 + 512);
        temper_tensor_destroy(&a);
        ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 512);
        temper_tensor_destroy(&b);
        ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 0);
        ASSERT(temper_scheduler_tier_logical(TEMPER_TIER_CPU) == 0);
    }
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_placement_score)
{
    temper_scheduler_init();
    TemperTensor t = make(256, 1.0f);
    TemperResource *r = t.resource;

    float same = temper_placement_score(r, TEMPER_DEVICE_CPU_0);
    float other = temper_placement_score(r, TEMPER_DEVICE_GPU_0);
    ASSERT(same > other); // no device-transfer penalty on the home device

    r->flags |= TEMPER_RESOURCE_RECOMPUTABLE;
    float recomputable_score = temper_placement_score(r, TEMPER_DEVICE_CPU_0);
    ASSERT(recomputable_score < same); // regenerable tensors place lower

    temper_tensor_destroy(&t);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_recompute_score)
{
    temper_scheduler_init();
    TemperTensor t = make(256, 1.0f);
    TemperResource *r = t.resource;

    ASSERT_FLOAT(temper_recompute_score(r), -1.0f, 1e-6f); // not recomputable

    r->flags |= TEMPER_RESOURCE_RECOMPUTABLE;
    r->access_count = 0;
    ASSERT_FLOAT(temper_recompute_score(r), 1.0f, 1e-6f);
    r->access_count = 1;
    ASSERT_FLOAT(temper_recompute_score(r), 0.5f, 1e-6f);
    r->access_count = 3;
    ASSERT_FLOAT(temper_recompute_score(r), 0.25f, 1e-6f);

    temper_tensor_destroy(&t);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_lru_eviction_demotes_to_compressed)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 512);
    temper_scheduler_set_tier_budget(TEMPER_TIER_COMPRESSED, 4096);

    TemperTensor t1 = make(256, 1.0f);
    TemperTensor t2 = make(256, 2.0f);
    // Deterministic access history: t1 is oldest.
    t1.resource->last_access = 100;
    t2.resource->last_access = 200;

    TemperTensor t3 = make(256, 3.0f); // pushes over budget -> evicts t1
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 256 + 256);
    ASSERT(t1.resource->flags & TEMPER_RESOURCE_COMPRESSED);
    ASSERT(t1.resource->tier == TEMPER_TIER_COMPRESSED);
    ASSERT(t1.resource->host_ptr == NULL);
    ASSERT(t1.resource->compressed_blob != NULL);
    ASSERT(t1.resource->compressed_size < t1.resource->bytes);
    ASSERT(t2.resource->tier == TEMPER_TIER_CPU);
    ASSERT(t3.resource->tier == TEMPER_TIER_CPU);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_COMPRESSED) == t1.resource->compressed_size);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_EVICTIONS) == 1);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_COMPRESSIONS) == 1);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_DEMOTIONS) == 1);

    temper_tensor_destroy(&t1);
    temper_tensor_destroy(&t2);
    temper_tensor_destroy(&t3);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_pinned_skipped_during_eviction)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 768);

    TemperTensor t1 = make(256, 1.0f);
    temper_tensor_pin(&t1);
    TemperTensor t2 = make(256, 2.0f);
    t1.resource->last_access = 100; // oldest, but pinned
    t2.resource->last_access = 200;

    ASSERT(temper_scheduler_tier_reserved(TEMPER_TIER_CPU) == 256);

    TemperTensor t3 = make(512, 3.0f); // 512 + 512 > 768 -> must evict 256
    ASSERT(t1.resource->flags & TEMPER_RESOURCE_RESIDENT); // pinned survived
    ASSERT(t1.resource->tier == TEMPER_TIER_CPU);
    ASSERT(t2.resource->flags & TEMPER_RESOURCE_COMPRESSED); // t2 evicted instead
    ASSERT(t2.resource->tier == TEMPER_TIER_COMPRESSED);

    temper_tensor_unpin(&t1);
    ASSERT(temper_scheduler_tier_reserved(TEMPER_TIER_CPU) == 0);

    temper_tensor_destroy(&t1);
    temper_tensor_destroy(&t2);
    temper_tensor_destroy(&t3);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_recompute_eviction_drops_storage)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 512);

    TemperTensor t1 = make(256, 1.0f);
    t1.resource->flags |= TEMPER_RESOURCE_RECOMPUTABLE;
    t1.resource->access_count = 0; // never accessed -> recompute wins
    t1.resource->last_access = 100;
    TemperTensor t2 = make(256, 2.0f);
    t2.resource->last_access = 200;

    size_t logical_before = temper_scheduler_tier_logical(TEMPER_TIER_CPU);
    TemperTensor t3 = make(256, 3.0f); // pushes over budget -> drops t1

    ASSERT(!(t1.resource->flags & TEMPER_RESOURCE_RESIDENT));
    ASSERT(t1.resource->host_ptr == NULL);
    ASSERT(t1.resource->compressed_blob == NULL);
    ASSERT(t1.resource->tier == TEMPER_TIER_CPU); // logical tier retained
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 256 + 256);
    ASSERT(temper_scheduler_tier_logical(TEMPER_TIER_CPU) == logical_before + 256);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_RECOMPUTATIONS) == 1);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_EVICTIONS) == 1);

    temper_tensor_destroy(&t1);
    temper_tensor_destroy(&t2);
    temper_tensor_destroy(&t3);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_dropped_resource_access_returns_null)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 512);

    TemperTensor t1 = make(256, 1.0f);
    t1.resource->flags |= TEMPER_RESOURCE_RECOMPUTABLE;
    t1.resource->access_count = 0;
    t1.resource->last_access = 100;
    TemperTensor t2 = make(256, 2.0f);
    t2.resource->last_access = 200;
    TemperTensor t3 = make(256, 3.0f);

    ASSERT(!(t1.resource->flags & TEMPER_RESOURCE_RESIDENT));
    ASSERT(temper_tensor_data(&t1) == NULL); // recompute replay is Phase 7
    ASSERT(temper_tensor_data(&t2) != NULL); // t2 unaffected

    temper_tensor_destroy(&t1);
    temper_tensor_destroy(&t2);
    temper_tensor_destroy(&t3);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_promote_on_access_decompresses)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_CPU, 512);
    temper_scheduler_set_tier_budget(TEMPER_TIER_COMPRESSED, 4096);

    TemperTensor t1 = make(256, 1.0f);
    t1.resource->last_access = 100;
    TemperTensor t2 = make(256, 2.0f);
    t2.resource->last_access = 200;
    TemperTensor t3 = make(256, 3.0f); // compresses t1

    ASSERT(t1.resource->flags & TEMPER_RESOURCE_COMPRESSED);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_COMPRESSED) > 0);

    float *data = temper_tensor_data(&t1); // access -> promote back to CPU
    ASSERT(data != NULL);
    ASSERT(!(t1.resource->flags & TEMPER_RESOURCE_COMPRESSED));
    ASSERT(t1.resource->tier == TEMPER_TIER_CPU);
    ASSERT(t1.resource->host_ptr == data);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_COMPRESSED) == 0);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_DECOMPRESSIONS) == 1);
    ASSERT(temper_scheduler_stat(TEMPER_STAT_PROMOTIONS) == 1);

    // 0.5-increment values are exactly bf16-representable.
    for (size_t i = 0; i < 16; i++)
    {
        ASSERT(ASSERT_FLOAT(data[i], 1.0f + (float)i * 0.5f, 1e-4f));
    }

    temper_tensor_destroy(&t1);
    temper_tensor_destroy(&t2);
    temper_tensor_destroy(&t3);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_can_compress_gate)
{
    temper_scheduler_init();

    // 8 floats = 32 bytes, below the bf16 minimum of 64 bytes.
    TemperShape s = temper_shape_1d(8);
    TemperTensor tiny = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ASSERT(tiny.resource != NULL);
    ASSERT(!temper_compression_bf16.can_compress(tiny.resource));
    ASSERT(temper_resource_demote(tiny.resource) != 0); // cannot demote
    ASSERT(tiny.resource->flags & TEMPER_RESOURCE_RESIDENT); // unchanged
    ASSERT(tiny.resource->tier == TEMPER_TIER_CPU);
    ASSERT(tiny.resource->compressed_blob == NULL);

    temper_tensor_destroy(&tiny);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

static bool stub_can_compress(const TemperResource *res)
{
    (void)res;
    return false;
}

static void *stub_compress(const float *src, size_t n, size_t *out_size)
{
    (void)src;
    (void)n;
    *out_size = 0;
    return NULL;
}

static float *stub_decompress(const void *blob, size_t bs, size_t n)
{
    (void)blob;
    (void)bs;
    (void)n;
    return NULL;
}

static size_t stub_estimate_size(size_t n)
{
    (void)n;
    return 0;
}

static const TemperCompressionBackend stub_backend = {
    "stub", stub_can_compress, stub_compress, stub_decompress, stub_estimate_size
};

TEST(test_compression_backend_swap)
{
    temper_scheduler_init();
    ASSERT(strcmp(temper_compression_bf16.name, "bf16") == 0);
    ASSERT(temper_scheduler_compression_backend() == &temper_compression_bf16);

    temper_scheduler_set_compression_backend(&stub_backend);
    ASSERT(temper_scheduler_compression_backend() == &stub_backend);

    TemperShape s = temper_shape_1d(64);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    t.resource->last_access = 100;
    ASSERT(temper_resource_demote(t.resource) != 0); // stub cannot compress
    ASSERT(t.resource->flags & TEMPER_RESOURCE_RESIDENT);

    temper_scheduler_set_compression_backend(&temper_compression_bf16);
    ASSERT(temper_scheduler_compression_backend() == &temper_compression_bf16);

    temper_tensor_destroy(&t);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

TEST(test_gpu_demote_to_cpu)
{
    temper_scheduler_init();
    temper_scheduler_set_tier_budget(TEMPER_TIER_GPU, 512);

    TemperShape s = temper_shape_1d(64);
    TemperTensor t = temper_tensor_create_on_device(s, TEMPER_DTYPE_F32, TEMPER_DEVICE_GPU_0);
    ASSERT(t.resource->tier == TEMPER_TIER_GPU);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_GPU) == 256);

    ASSERT(temper_resource_demote(t.resource) == 0);
    ASSERT(t.resource->tier == TEMPER_TIER_CPU);
    ASSERT(temper_device_equal(t.resource->device, TEMPER_DEVICE_CPU_0));
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_GPU) == 0);
    ASSERT(temper_scheduler_tier_used(TEMPER_TIER_CPU) == 256);

    temper_tensor_destroy(&t);
    ASSERT(temper_scheduler_validate());
    temper_scheduler_shutdown();
}

int main(void)
{
    printf("=== Memory Scheduler Tests ===\n");
    RUN(test_scheduler_init_shutdown);
    RUN(test_tier_budget_override);
    RUN(test_pressure_math);
    RUN(test_resource_register_tracking);
    RUN(test_placement_score);
    RUN(test_recompute_score);
    RUN(test_lru_eviction_demotes_to_compressed);
    RUN(test_pinned_skipped_during_eviction);
    RUN(test_recompute_eviction_drops_storage);
    RUN(test_dropped_resource_access_returns_null);
    RUN(test_promote_on_access_decompresses);
    RUN(test_can_compress_gate);
    RUN(test_compression_backend_swap);
    RUN(test_gpu_demote_to_cpu);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
