#include "temper/temper.h"
#include "temper/core/memory.h"
#include "temper/core/logger.h"
#include "temper/core/profiler.h"
#include "temper/core/platform.h"
#include "temper/core/threading.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
    #include <windows.h>
    typedef volatile LONG temper_atomic_int;
    #define temper_atomic_store(ptr, val) InterlockedExchange((ptr), (LONG)(val))
    #define temper_atomic_load(ptr) InterlockedCompareExchange((ptr), 0, 0)
    #define temper_atomic_fetch_add(ptr, val) InterlockedExchangeAdd((ptr), (LONG)(val))
#else
    #include <stdatomic.h>
    typedef _Atomic int temper_atomic_int;
    #define temper_atomic_store(ptr, val) atomic_store((ptr), (val))
    #define temper_atomic_load(ptr) atomic_load((ptr))
    #define temper_atomic_fetch_add(ptr, val) atomic_fetch_add((ptr), (val))
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

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

TEST(test_arena_create_destroy)
{
    TemperArena arena = temper_arena_create(1024);
    ASSERT(arena.buffer != NULL);
    ASSERT(arena.capacity == 1024);
    ASSERT(arena.offset == 0);
    temper_arena_destroy(&arena);
    ASSERT(arena.buffer == NULL);
}

TEST(test_arena_alloc)
{
    TemperArena arena = temper_arena_create(1024);
    void *p1 = temper_arena_alloc(&arena, 128);
    ASSERT(p1 != NULL);
    void *p2 = temper_arena_alloc(&arena, 256);
    ASSERT(p2 != NULL);
    ASSERT(p2 > p1);
    temper_arena_destroy(&arena);
}

TEST(test_arena_reset)
{
    TemperArena arena = temper_arena_create(1024);
    temper_arena_alloc(&arena, 128);
    temper_arena_reset(&arena);
    ASSERT(arena.offset == 0);
    temper_arena_destroy(&arena);
}

TEST(test_pool_create)
{
    TemperPool pool = temper_pool_create(64, 16);
    ASSERT(pool.buffer != NULL);
    ASSERT(pool.block_count == 16);
    ASSERT(pool.free_count == 16);
    temper_pool_destroy(&pool);
}

TEST(test_pool_alloc_free)
{
    TemperPool pool = temper_pool_create(64, 4);
    void *p1 = temper_pool_alloc(&pool);
    ASSERT(p1 != NULL);
    void *p2 = temper_pool_alloc(&pool);
    ASSERT(p2 != NULL);
    ASSERT(p1 != p2);
    temper_pool_free(&pool, p1);
    void *p3 = temper_pool_alloc(&pool);
    ASSERT(p3 == p1);
    temper_pool_destroy(&pool);
}

TEST(test_default_allocator)
{
    TemperAllocator a = temper_default_allocator();
    ASSERT(a.alloc != NULL);
    ASSERT(a.realloc != NULL);
    ASSERT(a.free != NULL);
    void *p = a.alloc(64, a.user_data);
    ASSERT(p != NULL);
    a.free(p, a.user_data);
}

TEST(test_logger)
{
    temper_log_set_level(TEMPER_LOG_TRACE);
    temper_info("Logger test message");
    ASSERT(1);
}

TEST(test_profiler)
{
    TemperProfiler prof;
    temper_profiler_init(&prof);
    temper_profiler_record_alloc(&prof, 100);
    ASSERT(prof.current_usage == 100);
    ASSERT(prof.peak_usage == 100);
    temper_profiler_record_free(&prof, 50);
    ASSERT(prof.current_usage == 50);
    temper_profiler_reset(&prof);
    ASSERT(prof.current_usage == 0);
}

TEST(test_platform)
{
    TemperPlatform p = temper_platform_get();
    ASSERT(p != TEMPER_PLATFORM_UNKNOWN);
    const char *name = temper_platform_name(p);
    ASSERT(name != NULL);
    uint64_t t = temper_time_us();
    ASSERT(t > 0);
}

// --- Thread Pool Tests ---

static temper_atomic_int g_counter = 0;

static void increment_job(void *data)
{
    (void)data;
    temper_atomic_fetch_add(&g_counter, 1);
}

TEST(test_thread_pool_create_destroy)
{
    TemperThreadPool *pool = temper_thread_pool_create(4);
    ASSERT(pool != NULL);
    temper_thread_pool_destroy(pool);
}

TEST(test_thread_pool_submit_and_wait)
{
    TemperThreadPool *pool = temper_thread_pool_create(2);
    ASSERT(pool != NULL);
    temper_atomic_store(&g_counter, 0);
    for (int i = 0; i < 10; i++)
    {
        int ret = temper_thread_pool_submit(pool, increment_job, NULL);
        ASSERT(ret == 0);
    }
    temper_thread_pool_wait(pool);
    ASSERT(temper_atomic_load(&g_counter) == 10);
    temper_thread_pool_destroy(pool);
}

static void sum_job(void *data)
{
    int *val = (int *)data;
    temper_atomic_fetch_add(&g_counter, *val);
}

TEST(test_thread_pool_parallel_sum)
{
    TemperThreadPool *pool = temper_thread_pool_create(4);
    temper_atomic_store(&g_counter, 0);
    int values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 8; i++)
    {
        temper_thread_pool_submit(pool, sum_job, &values[i]);
    }
    temper_thread_pool_wait(pool);
    ASSERT(temper_atomic_load(&g_counter) == 36);
    temper_thread_pool_destroy(pool);
}

static void slow_job(void *data)
{
    (void)data;
    temper_sleep_ms(10);
    temper_atomic_fetch_add(&g_counter, 1);
}

TEST(test_thread_pool_queue_full)
{
    TemperThreadPool *pool = temper_thread_pool_create(1);
    temper_atomic_store(&g_counter, 0);
    // Submit enough to fill queue (256 capacity) + overwhelm
    int full = 0;
    for (int i = 0; i < 300; i++)
    {
        int ret = temper_thread_pool_submit(pool, slow_job, NULL);
        if (ret == -1)
            full = 1;
    }
    temper_thread_pool_wait(pool);
    ASSERT(full);
    temper_thread_pool_destroy(pool);
}

// --- Memory Pool Stress Tests ---

TEST(test_pool_stress_alloc_free_cycle)
{
    TemperPool pool = temper_pool_create(64, 128);
    void *ptrs[128];
    // Allocate all blocks
    for (int i = 0; i < 128; i++)
    {
        ptrs[i] = temper_pool_alloc(&pool);
        ASSERT(ptrs[i] != NULL);
    }
    // Pool should be empty
    void *extra = temper_pool_alloc(&pool);
    ASSERT(extra == NULL);
    // Free all in reverse order
    for (int i = 127; i >= 0; i--)
    {
        temper_pool_free(&pool, ptrs[i]);
    }
    // Allocate again - should work
    for (int i = 0; i < 128; i++)
    {
        ptrs[i] = temper_pool_alloc(&pool);
        ASSERT(ptrs[i] != NULL);
    }
    temper_pool_destroy(&pool);
}

TEST(test_pool_interleaved)
{
    TemperPool pool = temper_pool_create(32, 16);
    void *a = temper_pool_alloc(&pool);
    void *b = temper_pool_alloc(&pool);
    temper_pool_free(&pool, a);
    void *c = temper_pool_alloc(&pool);
    ASSERT(c == a); // Should reuse freed block
    temper_pool_free(&pool, b);
    temper_pool_free(&pool, c);
    temper_pool_destroy(&pool);
}

TEST(test_arena_stress)
{
    TemperArena arena = temper_arena_create(4096);
    for (int i = 0; i < 100; i++)
    {
        void *p = temper_arena_alloc(&arena, 32);
        ASSERT(p != NULL);
    }
    ASSERT(arena.offset > 0);
    temper_arena_reset(&arena);
    ASSERT(arena.offset == 0);
    // Allocate again after reset
    void *p = temper_arena_alloc(&arena, 64);
    ASSERT(p != NULL);
    temper_arena_destroy(&arena);
}

TEST(test_arena_alignment)
{
    TemperArena arena = temper_arena_create(1024);
    void *p1 = temper_arena_alloc(&arena, 3);
    void *p2 = temper_arena_alloc(&arena, 7);
    // Check 8-byte alignment
    ASSERT(((uintptr_t)p1 % 8) == 0);
    ASSERT(((uintptr_t)p2 % 8) == 0);
    temper_arena_destroy(&arena);
}

int main(void)
{
    printf("=== Core Tests ===\n");
    RUN(test_arena_create_destroy);
    RUN(test_arena_alloc);
    RUN(test_arena_reset);
    RUN(test_pool_create);
    RUN(test_pool_alloc_free);
    RUN(test_default_allocator);
    RUN(test_logger);
    RUN(test_profiler);
    RUN(test_platform);
    RUN(test_thread_pool_create_destroy);
    RUN(test_thread_pool_submit_and_wait);
    RUN(test_thread_pool_parallel_sum);
    RUN(test_thread_pool_queue_full);
    RUN(test_pool_stress_alloc_free_cycle);
    RUN(test_pool_interleaved);
    RUN(test_arena_stress);
    RUN(test_arena_alignment);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
