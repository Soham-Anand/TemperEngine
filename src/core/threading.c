#include "temper/core/threading.h"
#include "temper/core/logger.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <stdatomic.h>
#endif

typedef struct Job
{
    TemperJobFunc func;
    void *data;
} Job;

#ifdef _WIN32

struct TemperThreadPool
{
    HANDLE *threads;
    uint32_t thread_count;
    CRITICAL_SECTION cs;
    HANDLE semaphore;
    Job *jobs;
    uint32_t job_capacity;
    uint32_t job_head;
    uint32_t job_tail;
    uint32_t job_count;
    uint32_t active_count;
    int shutdown;
};

static DWORD WINAPI worker_proc(LPVOID param)
{
    TemperThreadPool *pool = (TemperThreadPool *)param;
    while (1)
    {
        WaitForSingleObject(pool->semaphore, INFINITE);
        EnterCriticalSection(&pool->cs);
        if (pool->shutdown && pool->job_count == 0)
        {
            LeaveCriticalSection(&pool->cs);
            break;
        }
        Job job = pool->jobs[pool->job_head];
        pool->job_head = (pool->job_head + 1) % pool->job_capacity;
        pool->job_count--;
        pool->active_count++;
        LeaveCriticalSection(&pool->cs);
        job.func(job.data);
        EnterCriticalSection(&pool->cs);
        pool->active_count--;
        LeaveCriticalSection(&pool->cs);
    }
    return 0;
}

#else

struct TemperThreadPool
{
    pthread_t *threads;
    uint32_t thread_count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    Job *jobs;
    uint32_t job_capacity;
    uint32_t job_head;
    uint32_t job_tail;
    uint32_t job_count;
    uint32_t active_count;
    int shutdown;
    atomic_int done;
};

static void *worker_proc(void *param)
{
    TemperThreadPool *pool = (TemperThreadPool *)param;
    while (1)
    {
        pthread_mutex_lock(&pool->mutex);
        while (pool->job_count == 0 && !pool->shutdown)
        {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }
        if (pool->shutdown && pool->job_count == 0)
        {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        Job job = pool->jobs[pool->job_head];
        pool->job_head = (pool->job_head + 1) % pool->job_capacity;
        pool->job_count--;
        pool->active_count++;
        pthread_mutex_unlock(&pool->mutex);
        job.func(job.data);
        pthread_mutex_lock(&pool->mutex);
        pool->active_count--;
        if (pool->active_count == 0 && pool->job_count == 0)
        {
            pthread_cond_broadcast(&pool->cond);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

#endif

TemperThreadPool *temper_thread_pool_create(uint32_t thread_count)
{
    TemperThreadPool *pool = (TemperThreadPool *)calloc(1, sizeof(TemperThreadPool));
    pool->thread_count = thread_count;
    pool->job_capacity = 256;
    pool->jobs = (Job *)calloc(pool->job_capacity, sizeof(Job));

#ifdef _WIN32
    InitializeCriticalSection(&pool->cs);
    pool->semaphore = CreateSemaphore(NULL, 0, 10000, NULL);
    pool->threads = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
    for (uint32_t i = 0; i < thread_count; i++)
    {
        pool->threads[i] = CreateThread(NULL, 0, worker_proc, pool, 0, NULL);
    }
#else
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    pool->threads = (pthread_t *)calloc(thread_count, sizeof(pthread_t));
    for (uint32_t i = 0; i < thread_count; i++)
    {
        pthread_create(&pool->threads[i], NULL, worker_proc, pool);
    }
#endif

    temper_info("Thread pool created with %u threads", thread_count);
    return pool;
}

void temper_thread_pool_destroy(TemperThreadPool *pool)
{
    if (!pool)
    {
        return;
    }

#ifdef _WIN32
    EnterCriticalSection(&pool->cs);
    pool->shutdown = 1;
    LeaveCriticalSection(&pool->cs);
    ReleaseSemaphore(pool->semaphore, pool->thread_count, NULL);
    WaitForMultipleObjects(pool->thread_count, pool->threads, TRUE, INFINITE);
    for (uint32_t i = 0; i < pool->thread_count; i++)
    {
        CloseHandle(pool->threads[i]);
    }
    CloseHandle(pool->semaphore);
    DeleteCriticalSection(&pool->cs);
#else
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t i = 0; i < pool->thread_count; i++)
    {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
#endif

    free(pool->threads);
    free(pool->jobs);
    free(pool);
}

int temper_thread_pool_submit(TemperThreadPool *pool, TemperJobFunc func, void *data)
{
#ifdef _WIN32
    EnterCriticalSection(&pool->cs);
    if (pool->job_count >= pool->job_capacity)
    {
        LeaveCriticalSection(&pool->cs);
        return -1;
    }
    pool->jobs[pool->job_tail] = (Job){func, data};
    pool->job_tail = (pool->job_tail + 1) % pool->job_capacity;
    pool->job_count++;
    LeaveCriticalSection(&pool->cs);
    ReleaseSemaphore(pool->semaphore, 1, NULL);
#else
    pthread_mutex_lock(&pool->mutex);
    if (pool->job_count >= pool->job_capacity)
    {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }
    pool->jobs[pool->job_tail] = (Job){func, data};
    pool->job_tail = (pool->job_tail + 1) % pool->job_capacity;
    pool->job_count++;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
#endif
    return 0;
}

void temper_thread_pool_wait(TemperThreadPool *pool)
{
#ifdef _WIN32
    EnterCriticalSection(&pool->cs);
    while (pool->job_count > 0 || pool->active_count > 0)
    {
        LeaveCriticalSection(&pool->cs);
        Sleep(1);
        EnterCriticalSection(&pool->cs);
    }
    LeaveCriticalSection(&pool->cs);
#else
    pthread_mutex_lock(&pool->mutex);
    while (pool->job_count > 0 || pool->active_count > 0)
    {
        pthread_cond_wait(&pool->cond, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
#endif
}
