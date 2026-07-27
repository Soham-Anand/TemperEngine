#include "temper/temper.h"
#include "temper/core/memory.h"
#include "temper/core/logger.h"
#include "temper/core/profiler.h"
#include "temper/core/platform.h"
#include "temper/utils/assert.h"
#include <stdio.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) static void name(void)
#define RUN(name)                                                             \
    do                                                                        \
    {                                                                         \
        tests_run++;                                                          \
        printf("  %-40s", #name);                                             \
        name();                                                               \
        tests_passed++;                                                       \
        printf("PASS\n");                                                     \
    } while (0)

#define ASSERT(cond)                                                          \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
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
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
