#ifndef TEMPER_MEMORY_H
#define TEMPER_MEMORY_H

#include <stddef.h>

typedef struct TemperAllocator
{
    void *(*alloc)(size_t size, void *user_data);
    void *(*realloc)(void *ptr, size_t size, void *user_data);
    void (*free)(void *ptr, void *user_data);
    void *user_data;
} TemperAllocator;

typedef struct TemperArena
{
    char *buffer;
    size_t capacity;
    size_t offset;
} TemperArena;

typedef struct TemperPool
{
    char *buffer;
    size_t block_size;
    size_t block_count;
    size_t *free_list;
    size_t free_count;
} TemperPool;

TemperAllocator temper_default_allocator(void);
TemperAllocator temper_arena_allocator(TemperArena *arena);

TemperArena temper_arena_create(size_t capacity);
void temper_arena_destroy(TemperArena *arena);
void *temper_arena_alloc(TemperArena *arena, size_t size);
void temper_arena_reset(TemperArena *arena);

TemperPool temper_pool_create(size_t block_size, size_t block_count);
void temper_pool_destroy(TemperPool *pool);
void *temper_pool_alloc(TemperPool *pool);
void temper_pool_free(TemperPool *pool, void *ptr);

#endif
