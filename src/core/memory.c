#include "temper/core/memory.h"
#include "temper/utils/assert.h"
#include <stdlib.h>
#include <string.h>

static void *default_alloc(size_t size, void *ud)
{
    (void)ud;
    return malloc(size);
}

static void *default_realloc(void *ptr, size_t size, void *ud)
{
    (void)ud;
    return realloc(ptr, size);
}

static void default_free(void *ptr, void *ud)
{
    (void)ud;
    free(ptr);
}

TemperAllocator temper_default_allocator(void)
{
    TemperAllocator a = {0};
    a.alloc = default_alloc;
    a.realloc = default_realloc;
    a.free = default_free;
    a.user_data = NULL;
    return a;
}

static void *arena_alloc_fn(size_t size, void *ud)
{
    TemperArena *arena = (TemperArena *)ud;
    return temper_arena_alloc(arena, size);
}

static void *arena_realloc_fn(void *ptr, size_t size, void *ud)
{
    (void)ptr;
    (void)ud;
    (void)size;
    return NULL; // Arenas do not support realloc
}

static void arena_free_fn(void *ptr, void *ud)
{
    (void)ptr;
    (void)ud;
    // No-op: arena memory is freed as a whole
}

TemperAllocator temper_arena_allocator(TemperArena *arena)
{
    TemperAllocator a = {0};
    a.alloc = arena_alloc_fn;
    a.realloc = arena_realloc_fn;
    a.free = arena_free_fn;
    a.user_data = arena;
    return a;
}

TemperArena temper_arena_create(size_t capacity)
{
    TemperArena arena = {0};
    arena.buffer = (char *)malloc(capacity);
    TEMPER_ASSERT_MSG(arena.buffer != NULL, "Arena allocation failed");
    arena.capacity = capacity;
    arena.offset = 0;
    return arena;
}

void temper_arena_destroy(TemperArena *arena)
{
    if (arena->buffer)
    {
        free(arena->buffer);
        arena->buffer = NULL;
        arena->capacity = 0;
        arena->offset = 0;
    }
}

void *temper_arena_alloc(TemperArena *arena, size_t size)
{
    size_t aligned = (size + 7) & ~(size_t)7;
    if (arena->offset + aligned > arena->capacity)
    {
        return NULL;
    }
    void *ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

void temper_arena_reset(TemperArena *arena)
{
    arena->offset = 0;
}

TemperPool temper_pool_create(size_t block_size, size_t block_count)
{
    TemperPool pool = {0};
    size_t aligned = (block_size + 7) & ~(size_t)7;
    pool.buffer = (char *)malloc(aligned * block_count);
    TEMPER_ASSERT_MSG(pool.buffer != NULL, "Pool allocation failed");
    pool.block_size = aligned;
    pool.block_count = block_count;
    pool.free_list = (size_t *)malloc(sizeof(size_t) * block_count);
    pool.free_count = block_count;
    for (size_t i = 0; i < block_count; i++)
    {
        pool.free_list[i] = i;
    }
    return pool;
}

void temper_pool_destroy(TemperPool *pool)
{
    if (pool->buffer)
    {
        free(pool->buffer);
        free(pool->free_list);
        pool->buffer = NULL;
        pool->free_list = NULL;
        pool->free_count = 0;
    }
}

void *temper_pool_alloc(TemperPool *pool)
{
    if (pool->free_count == 0)
    {
        return NULL;
    }
    size_t idx = pool->free_list[--pool->free_count];
    return pool->buffer + idx * pool->block_size;
}

void temper_pool_free(TemperPool *pool, void *ptr)
{
    char *p = (char *)ptr;
    size_t idx = (size_t)(p - pool->buffer) / pool->block_size;
    TEMPER_ASSERT(idx < pool->block_count);
    pool->free_list[pool->free_count++] = idx;
}
