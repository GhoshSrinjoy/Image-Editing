/* Minimal parallel-for used by the image operations. */
#ifndef IE_PARALLEL_H
#define IE_PARALLEL_H

typedef void (*ie_range_fn)(void *ctx, int begin, int end);

/* Calls fn(ctx, begin, end) over disjoint chunks covering [0, n), spread over
 * the worker threads. Chunks are at least `grain` long. Nested calls (from
 * inside fn) run inline on the calling thread. */
void ie_parallel_for(int n, int grain, ie_range_fn fn, void *ctx);

int ie_cpu_count(void);

#endif
