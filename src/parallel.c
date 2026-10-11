/*
 * Parallel-for over a persistent worker pool. Workers are started on first
 * use and sleep on a condition variable between jobs, so the per-call cost is
 * a wake-up rather than thread creation. Work is handed out in chunks through
 * an atomic counter, so fast cores take more of it.
 */
#include "parallel.h"
#include "imgedit.h"

#include <stdlib.h>

#define IE_MAX_THREADS 64

#if defined(__wasm__) || defined(IE_NO_THREADS)
#define IE_SINGLE_THREAD 1
#elif defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(IE_SINGLE_THREAD)
#define IE_TLS
#elif defined(_MSC_VER)
#define IE_TLS __declspec(thread)
#else
#define IE_TLS _Thread_local
#endif

static int g_threads; /* 0 = auto */

void ie_set_threads(int n) { g_threads = n < 0 ? 0 : n; }

int ie_get_threads(void)
{
    int n = g_threads > 0 ? g_threads : ie_cpu_count();
    return n > IE_MAX_THREADS ? IE_MAX_THREADS : n;
}

#if defined(IE_SINGLE_THREAD)

int ie_cpu_count(void) { return 1; }

void ie_parallel_for(int n, int grain, ie_range_fn fn, void *ctx)
{
    (void)grain;
    if (n > 0) fn(ctx, 0, n);
}

#else

static IE_TLS int t_nested; /* set on pool threads and while a job runs */

typedef struct {
    ie_range_fn fn;
    void *ctx;
    int n, chunk, max_workers;
    volatile long next; /* first index not yet handed out */
} job;

#if defined(_WIN32)
int ie_cpu_count(void)
{
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n > 0 ? (int)n : 1;
}
#define ATOMIC_FETCH_ADD(p, v) InterlockedExchangeAdd((volatile LONG *)(p), (LONG)(v))
static SRWLOCK lock = SRWLOCK_INIT, submit = SRWLOCK_INIT;
static CONDITION_VARIABLE wake = CONDITION_VARIABLE_INIT, done = CONDITION_VARIABLE_INIT;
#define LOCK(l) AcquireSRWLockExclusive(&l)
#define UNLOCK(l) ReleaseSRWLockExclusive(&l)
#define WAIT(cv) SleepConditionVariableSRW(&cv, &lock, INFINITE, 0)
#define BROADCAST(cv) WakeAllConditionVariable(&cv)
#define SIGNAL(cv) WakeConditionVariable(&cv)
#else
int ie_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}
#define ATOMIC_FETCH_ADD(p, v) __atomic_fetch_add((p), (v), __ATOMIC_RELAXED)
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER, submit = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wake = PTHREAD_COND_INITIALIZER, done = PTHREAD_COND_INITIALIZER;
#define LOCK(l) pthread_mutex_lock(&l)
#define UNLOCK(l) pthread_mutex_unlock(&l)
#define WAIT(cv) pthread_cond_wait(&cv, &lock)
#define BROADCAST(cv) pthread_cond_broadcast(&cv)
#define SIGNAL(cv) pthread_cond_signal(&cv)
#endif

/* Pool state, guarded by `lock`. */
static job *cur_job;
static unsigned long generation;
static int nworkers, pending;

static void job_run(job *j)
{
    for (;;) {
        long b = ATOMIC_FETCH_ADD(&j->next, j->chunk);
        if (b >= j->n) break;
        long e = b + j->chunk;
        j->fn(j->ctx, (int)b, (int)(e > j->n ? j->n : e));
    }
}

typedef struct {
    int index;
    unsigned long seen;
} worker_arg;

static void worker_loop(worker_arg a)
{
    t_nested = 1;
    LOCK(lock);
    for (;;) {
        while (generation == a.seen) WAIT(wake);
        a.seen = generation;
        job *j = cur_job;
        UNLOCK(lock);
        if (a.index < j->max_workers) job_run(j);
        LOCK(lock);
        if (--pending == 0) SIGNAL(done);
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker_main(LPVOID p)
{
    worker_arg a = *(worker_arg *)p;
    free(p);
    worker_loop(a);
    return 0;
}
#else
static void *worker_main(void *p)
{
    worker_arg a = *(worker_arg *)p;
    free(p);
    worker_loop(a);
    return NULL;
}
#endif

/* Called with `lock` held. New workers start at the current generation, so
 * they join the next job rather than an old one. */
static void grow_pool(int want)
{
    while (nworkers < want) {
        worker_arg *a = (worker_arg *)malloc(sizeof *a);
        if (!a) return;
        a->index = nworkers;
        a->seen = generation;
#if defined(_WIN32)
        HANDLE h = CreateThread(NULL, 0, worker_main, a, 0, NULL);
        if (!h) {
            free(a);
            return;
        }
        CloseHandle(h);
#else
        pthread_t t;
        if (pthread_create(&t, NULL, worker_main, a) != 0) {
            free(a);
            return;
        }
        pthread_detach(t);
#endif
        nworkers++;
    }
}

void ie_parallel_for(int n, int grain, ie_range_fn fn, void *ctx)
{
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    int nt = ie_get_threads();
    int max_chunks = (n + grain - 1) / grain;
    if (nt > max_chunks) nt = max_chunks;
    if (nt <= 1 || t_nested) {
        fn(ctx, 0, n);
        return;
    }

    job j;
    j.fn = fn;
    j.ctx = ctx;
    j.n = n;
    j.chunk = n / (nt * 4); /* several chunks per thread for load balance */
    if (j.chunk < grain) j.chunk = grain;
    j.next = 0;
    j.max_workers = nt - 1;

    LOCK(submit); /* one job in the pool at a time */
    LOCK(lock);
    grow_pool(nt - 1);
    cur_job = &j;
    pending = nworkers;
    generation++;
    BROADCAST(wake);
    UNLOCK(lock);

    /* The caller works too, and finishes everything if no worker started. */
    t_nested = 1;
    job_run(&j);
    t_nested = 0;

    LOCK(lock);
    while (pending > 0) WAIT(done);
    UNLOCK(lock);
    UNLOCK(submit);
}

#endif
