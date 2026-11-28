/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_INDEX_HEAP_H
#define NANOVEC_INDEX_HEAP_H

#include "core/alloc.h"
#include "core/types.h"

#include <math.h>

typedef enum nanovec_heap_mode {
    NANOVEC_HEAP_MAX     = 0,
    NANOVEC_HEAP_MIN     = 1,
    NANOVEC_HEAP_MIN_MAX = 2
} nanovec_heap_mode_t;

typedef struct nanovec_heap_item {
    nanovec_vid_t  vid;
    nanovec_dist_t dist;
    uint32_t       _pad;
} nanovec_heap_item_t;

NANOVEC_STATIC_ASSERT(sizeof(nanovec_heap_item_t) == 16U, "heap item must be 16 bytes");

typedef struct nanovec_heap {
    NANOVEC_ALIGNAS(NANOVEC_CACHE_LINE) nanovec_heap_item_t *elems;
    size_t               size;
    size_t               capacity;
    nanovec_heap_mode_t  mode;
    bool                 owns_mem;
} nanovec_heap_t;

NANOVEC_STATIC_ASSERT(_Alignof(nanovec_heap_t) >= NANOVEC_CACHE_LINE, "heap header must be 64B aligned");
NANOVEC_STATIC_ASSERT(sizeof(nanovec_heap_t) == NANOVEC_CACHE_LINE, "heap header must fit in 1 cache line");

/* Canonical total ordering: smaller distance is closer; ties broken by smaller VID. */
static inline bool nanovec_heap_item_lt(nanovec_heap_item_t a, nanovec_heap_item_t b) {
    if (a.dist < b.dist) return true;
    if (a.dist > b.dist) return false;
    return a.vid < b.vid;
}

static inline size_t nanovec_heap_size(const nanovec_heap_t *heap) {
    return (heap != NULL) ? heap->size : 0U;
}

static inline size_t nanovec_heap_capacity(const nanovec_heap_t *heap) {
    return (heap != NULL) ? heap->capacity : 0U;
}

static inline bool nanovec_heap_empty(const nanovec_heap_t *heap) {
    return heap == NULL || heap->size == 0U;
}

static inline bool nanovec_heap_full(const nanovec_heap_t *heap) {
    return heap != NULL && heap->size >= heap->capacity;
}

/*
 * Inline distance threshold for bounded top-K scans: rejects distances strictly
 * worse than current worst without entering push(). Returns +INFINITY until full.
 */
static inline nanovec_dist_t nanovec_heap_threshold(const nanovec_heap_t *heap) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->capacity == 0U)) {
        return -INFINITY;
    }
    if (heap->size < heap->capacity) {
        return INFINITY;
    }
    if (heap->mode == NANOVEC_HEAP_MAX) {
        return heap->elems[0].dist;
    }
    if (heap->mode == NANOVEC_HEAP_MIN_MAX) {
        if (heap->size == 1U) {
            return heap->elems[0].dist;
        }
        if (heap->size == 2U) {
            return heap->elems[1].dist;
        }
        return (heap->elems[1].dist >= heap->elems[2].dist) ? heap->elems[1].dist : heap->elems[2].dist;
    }
    return INFINITY;
}

nanovec_status_t nanovec_heap_init(
    nanovec_heap_t     *heap,
    size_t              capacity,
    nanovec_heap_mode_t mode
);

nanovec_status_t nanovec_heap_init_arena(
    nanovec_heap_t     *heap,
    nanovec_arena_t    *arena,
    size_t              capacity,
    nanovec_heap_mode_t mode
);

nanovec_status_t nanovec_heap_init_buffer(
    nanovec_heap_t      *heap,
    nanovec_heap_item_t *buffer,
    size_t               capacity,
    nanovec_heap_mode_t  mode
);

void nanovec_heap_destroy(nanovec_heap_t *heap);
void nanovec_heap_reset(nanovec_heap_t *heap);

bool nanovec_heap_push(nanovec_heap_t *heap, nanovec_vid_t vid, nanovec_dist_t dist);

bool nanovec_heap_peek(const nanovec_heap_t *heap, nanovec_heap_item_t *out);
bool nanovec_heap_peek_min(const nanovec_heap_t *heap, nanovec_heap_item_t *out);
bool nanovec_heap_peek_max(const nanovec_heap_t *heap, nanovec_heap_item_t *out);

bool nanovec_heap_pop(nanovec_heap_t *heap, nanovec_heap_item_t *out);
bool nanovec_heap_pop_min(nanovec_heap_t *heap, nanovec_heap_item_t *out);
bool nanovec_heap_pop_max(nanovec_heap_t *heap, nanovec_heap_item_t *out);

void nanovec_heap_sort_asc(nanovec_heap_t *heap);

size_t nanovec_heap_to_results(
    nanovec_heap_t   *heap,
    nanovec_result_t *out_results,
    size_t            max_results
);

size_t nanovec_heap_merge(nanovec_heap_t *dst, const nanovec_heap_t *src);

#endif /* NANOVEC_INDEX_HEAP_H */
