#ifndef TEMPER_THREADING_H
#define TEMPER_THREADING_H

#include <stdint.h>

typedef struct TemperThreadPool TemperThreadPool;
typedef void (*TemperJobFunc)(void *data);

TemperThreadPool *temper_thread_pool_create(uint32_t thread_count);
void temper_thread_pool_destroy(TemperThreadPool *pool);
int temper_thread_pool_submit(TemperThreadPool *pool, TemperJobFunc func, void *data);
void temper_thread_pool_wait(TemperThreadPool *pool);

#endif
