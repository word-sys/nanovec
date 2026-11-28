/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "index/heap.h"
#include "core/bitops.h"
#include "core/error.h"

#include <string.h>

static inline void nanovec_heap_swap(nanovec_heap_item_t *a, nanovec_heap_item_t *b) {
    nanovec_heap_item_t tmp = *a;
    *a                      = *b;
    *b                      = tmp;
}

static inline bool nanovec_heap_item_gt(nanovec_heap_item_t a, nanovec_heap_item_t b) {
    return nanovec_heap_item_lt(b, a);
}

static inline bool nanovec_minmax_is_min_level(size_t idx) {
    /* 0-indexed depth = 63 - clz64(idx + 1); even depth <=> odd clz64(idx + 1). */
    uint32_t lz = nanovec_clz64((uint64_t)idx + 1ULL);
    return (lz & 1U) != 0U;
}

static void nanovec_heap_sift_up_min(nanovec_heap_item_t *elems, size_t idx, nanovec_heap_item_t item) {
    while (idx > 0U) {
        size_t parent = (idx - 1U) >> 1U;
        if (!nanovec_heap_item_lt(item, elems[parent])) {
            break;
        }
        elems[idx] = elems[parent];
        idx        = parent;
    }
    elems[idx] = item;
}

static void nanovec_heap_sift_down_min(nanovec_heap_item_t *elems, size_t n, nanovec_heap_item_t item) {
    size_t idx  = 0U;
    size_t half = n >> 1U;
    while (idx < half) {
        size_t left  = (idx << 1U) + 1U;
        size_t right = left + 1U;
        size_t best  = left;
        if (right < n && nanovec_heap_item_lt(elems[right], elems[left])) {
            best = right;
        }
        if (!nanovec_heap_item_lt(elems[best], item)) {
            break;
        }
        elems[idx] = elems[best];
        idx        = best;
    }
    elems[idx] = item;
}

static void nanovec_heap_sift_up_max(nanovec_heap_item_t *elems, size_t idx, nanovec_heap_item_t item) {
    while (idx > 0U) {
        size_t parent = (idx - 1U) >> 1U;
        if (!nanovec_heap_item_gt(item, elems[parent])) {
            break;
        }
        elems[idx] = elems[parent];
        idx        = parent;
    }
    elems[idx] = item;
}

static void nanovec_heap_sift_down_max(nanovec_heap_item_t *elems, size_t n, nanovec_heap_item_t item) {
    size_t idx  = 0U;
    size_t half = n >> 1U;
    while (idx < half) {
        size_t left  = (idx << 1U) + 1U;
        size_t right = left + 1U;
        size_t best  = left;
        if (right < n && nanovec_heap_item_gt(elems[right], elems[left])) {
            best = right;
        }
        if (!nanovec_heap_item_gt(elems[best], item)) {
            break;
        }
        elems[idx] = elems[best];
        idx        = best;
    }
    elems[idx] = item;
}

static void nanovec_minmax_sift_up_min(nanovec_heap_item_t *elems, size_t idx) {
    while (idx >= 3U) {
        size_t gp = (idx - 3U) >> 2U;
        if (!nanovec_heap_item_lt(elems[idx], elems[gp])) {
            break;
        }
        nanovec_heap_swap(&elems[idx], &elems[gp]);
        idx = gp;
    }
}

static void nanovec_minmax_sift_up_max(nanovec_heap_item_t *elems, size_t idx) {
    while (idx >= 3U) {
        size_t gp = (idx - 3U) >> 2U;
        if (!nanovec_heap_item_gt(elems[idx], elems[gp])) {
            break;
        }
        nanovec_heap_swap(&elems[idx], &elems[gp]);
        idx = gp;
    }
}

static void nanovec_minmax_sift_up(nanovec_heap_item_t *elems, size_t idx) {
    if (idx == 0U) {
        return;
    }
    size_t parent = (idx - 1U) >> 1U;
    if (nanovec_minmax_is_min_level(idx)) {
        if (nanovec_heap_item_gt(elems[idx], elems[parent])) {
            nanovec_heap_swap(&elems[idx], &elems[parent]);
            nanovec_minmax_sift_up_max(elems, parent);
        } else {
            nanovec_minmax_sift_up_min(elems, idx);
        }
    } else {
        if (nanovec_heap_item_lt(elems[idx], elems[parent])) {
            nanovec_heap_swap(&elems[idx], &elems[parent]);
            nanovec_minmax_sift_up_min(elems, parent);
        } else {
            nanovec_minmax_sift_up_max(elems, idx);
        }
    }
}

static void nanovec_minmax_trickle_down_min(nanovec_heap_item_t *elems, size_t n, size_t idx) {
    for (;;) {
        size_t c1 = (idx << 1U) + 1U;
        if (c1 >= n) {
            break;
        }
        size_t c2 = c1 + 1U;
        size_t m  = c1;
        if (c2 < n && nanovec_heap_item_lt(elems[c2], elems[m])) {
            m = c2;
        }

        size_t g_start = (c1 << 1U) + 1U;
        size_t g_end   = (c2 < n) ? ((c2 << 1U) + 3U) : (g_start + 2U);
        if (g_end > n) {
            g_end = n;
        }
        for (size_t g = g_start; g < g_end; ++g) {
            /* Prefer grandchild on ties so a parent with children is never stopped prematurely. */
            if (!nanovec_heap_item_lt(elems[m], elems[g])) {
                m = g;
            }
        }

        if (m >= g_start) {
            if (nanovec_heap_item_lt(elems[m], elems[idx])) {
                nanovec_heap_swap(&elems[m], &elems[idx]);
                size_t p = (m - 1U) >> 1U;
                if (nanovec_heap_item_gt(elems[m], elems[p])) {
                    nanovec_heap_swap(&elems[m], &elems[p]);
                }
                idx = m;
                continue;
            }
            break;
        } else {
            if (nanovec_heap_item_lt(elems[m], elems[idx])) {
                nanovec_heap_swap(&elems[m], &elems[idx]);
            }
            break;
        }
    }
}

static void nanovec_minmax_trickle_down_max(nanovec_heap_item_t *elems, size_t n, size_t idx) {
    for (;;) {
        size_t c1 = (idx << 1U) + 1U;
        if (c1 >= n) {
            break;
        }
        size_t c2 = c1 + 1U;
        size_t m  = c1;
        if (c2 < n && nanovec_heap_item_gt(elems[c2], elems[m])) {
            m = c2;
        }

        size_t g_start = (c1 << 1U) + 1U;
        size_t g_end   = (c2 < n) ? ((c2 << 1U) + 3U) : (g_start + 2U);
        if (g_end > n) {
            g_end = n;
        }
        for (size_t g = g_start; g < g_end; ++g) {
            /* Prefer grandchild on ties so a parent with children is never stopped prematurely. */
            if (!nanovec_heap_item_gt(elems[m], elems[g])) {
                m = g;
            }
        }

        if (m >= g_start) {
            if (nanovec_heap_item_gt(elems[m], elems[idx])) {
                nanovec_heap_swap(&elems[m], &elems[idx]);
                size_t p = (m - 1U) >> 1U;
                if (nanovec_heap_item_lt(elems[m], elems[p])) {
                    nanovec_heap_swap(&elems[m], &elems[p]);
                }
                idx = m;
                continue;
            }
            break;
        } else {
            if (nanovec_heap_item_gt(elems[m], elems[idx])) {
                nanovec_heap_swap(&elems[m], &elems[idx]);
            }
            break;
        }
    }
}

static inline size_t nanovec_minmax_max_index(const nanovec_heap_t *heap) {
    if (heap->size == 1U) {
        return 0U;
    }
    if (heap->size == 2U) {
        return 1U;
    }
    return nanovec_heap_item_gt(heap->elems[1], heap->elems[2]) ? 1U : 2U;
}

static bool nanovec_heap_mode_valid(nanovec_heap_mode_t mode) {
    return mode == NANOVEC_HEAP_MAX || mode == NANOVEC_HEAP_MIN || mode == NANOVEC_HEAP_MIN_MAX;
}

nanovec_status_t nanovec_heap_init(
    nanovec_heap_t     *heap,
    size_t              capacity,
    nanovec_heap_mode_t mode
) {
    if (NANOVEC_UNLIKELY(heap == NULL || !nanovec_heap_mode_valid(mode))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid heap_init args");
    }

    memset(heap, 0, sizeof(*heap));
    heap->mode = mode;

    if (capacity == 0U) {
        return NANOVEC_SUCCESS;
    }
    if (NANOVEC_UNLIKELY(capacity > SIZE_MAX / sizeof(nanovec_heap_item_t))) {
        return NANOVEC_ERR_OOM;
    }

    heap->elems = (nanovec_heap_item_t *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE, capacity * sizeof(nanovec_heap_item_t)
    );
    if (NANOVEC_UNLIKELY(heap->elems == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    heap->capacity = capacity;
    heap->owns_mem = true;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_heap_init_arena(
    nanovec_heap_t     *heap,
    nanovec_arena_t    *arena,
    size_t              capacity,
    nanovec_heap_mode_t mode
) {
    if (NANOVEC_UNLIKELY(heap == NULL || arena == NULL || !nanovec_heap_mode_valid(mode))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid heap_init_arena args"
        );
    }

    memset(heap, 0, sizeof(*heap));
    heap->mode = mode;

    if (capacity == 0U) {
        return NANOVEC_SUCCESS;
    }
    if (NANOVEC_UNLIKELY(capacity > SIZE_MAX / sizeof(nanovec_heap_item_t))) {
        return NANOVEC_ERR_OOM;
    }

    heap->elems = (nanovec_heap_item_t *)nanovec_arena_alloc(
        arena, capacity * sizeof(nanovec_heap_item_t), NANOVEC_CACHE_LINE
    );
    if (NANOVEC_UNLIKELY(heap->elems == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    heap->capacity = capacity;
    heap->owns_mem = false;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_heap_init_buffer(
    nanovec_heap_t      *heap,
    nanovec_heap_item_t *buffer,
    size_t               capacity,
    nanovec_heap_mode_t  mode
) {
    if (NANOVEC_UNLIKELY(
            heap == NULL || !nanovec_heap_mode_valid(mode) || (capacity > 0U && buffer == NULL)
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid heap_init_buffer args"
        );
    }

    memset(heap, 0, sizeof(*heap));
    heap->elems    = (capacity > 0U) ? buffer : NULL;
    heap->capacity = capacity;
    heap->mode     = mode;
    heap->owns_mem = false;
    return NANOVEC_SUCCESS;
}

void nanovec_heap_destroy(nanovec_heap_t *heap) {
    if (heap == NULL) {
        return;
    }
    if (heap->owns_mem && heap->elems != NULL) {
        nanovec_aligned_free(heap->elems);
    }
    memset(heap, 0, sizeof(*heap));
}

void nanovec_heap_reset(nanovec_heap_t *heap) {
    if (heap != NULL) {
        heap->size = 0U;
    }
}

bool nanovec_heap_push(nanovec_heap_t *heap, nanovec_vid_t vid, nanovec_dist_t dist) {
    if (NANOVEC_UNLIKELY(
            heap == NULL || heap->elems == NULL || heap->capacity == 0U || isnan(dist)
        )) {
        return false;
    }

    nanovec_heap_item_t item;
    item.vid  = vid;
    item.dist = dist;
    item._pad = 0U;

    if (heap->size < heap->capacity) {
        size_t idx = heap->size++;
        switch (heap->mode) {
            case NANOVEC_HEAP_MAX:
                nanovec_heap_sift_up_max(heap->elems, idx, item);
                return true;
            case NANOVEC_HEAP_MIN:
                nanovec_heap_sift_up_min(heap->elems, idx, item);
                return true;
            case NANOVEC_HEAP_MIN_MAX:
                heap->elems[idx] = item;
                nanovec_minmax_sift_up(heap->elems, idx);
                return true;
        }
        return false;
    }

    switch (heap->mode) {
        case NANOVEC_HEAP_MAX:
            if (!nanovec_heap_item_lt(item, heap->elems[0])) {
                return false;
            }
            nanovec_heap_sift_down_max(heap->elems, heap->size, item);
            return true;

        case NANOVEC_HEAP_MIN:
            return false;

        case NANOVEC_HEAP_MIN_MAX: {
            size_t max_idx = nanovec_minmax_max_index(heap);
            if (!nanovec_heap_item_lt(item, heap->elems[max_idx])) {
                return false;
            }
            if (heap->size == 1U) {
                heap->elems[0] = item;
                return true;
            }
            /* If new candidate beats current global minimum, swap through root to keep min <= all. */
            if (nanovec_heap_item_lt(item, heap->elems[0])) {
                nanovec_heap_item_t old_min = heap->elems[0];
                heap->elems[0]              = item;
                heap->elems[max_idx]        = old_min;
            } else {
                heap->elems[max_idx] = item;
            }
            nanovec_minmax_trickle_down_max(heap->elems, heap->size, max_idx);
            return true;
        }
    }
    return false;
}

bool nanovec_heap_peek(const nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size == 0U)) {
        return false;
    }
    if (out != NULL) {
        *out = heap->elems[0];
    }
    return true;
}

bool nanovec_heap_peek_min(const nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size == 0U)) {
        return false;
    }
    if (heap->mode == NANOVEC_HEAP_MIN || heap->mode == NANOVEC_HEAP_MIN_MAX) {
        if (out != NULL) {
            *out = heap->elems[0];
        }
        return true;
    }
    return false;
}

bool nanovec_heap_peek_max(const nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size == 0U)) {
        return false;
    }
    if (heap->mode == NANOVEC_HEAP_MAX) {
        if (out != NULL) {
            *out = heap->elems[0];
        }
        return true;
    }
    if (heap->mode == NANOVEC_HEAP_MIN_MAX) {
        size_t max_idx = nanovec_minmax_max_index(heap);
        if (out != NULL) {
            *out = heap->elems[max_idx];
        }
        return true;
    }
    return false;
}

bool nanovec_heap_pop_min(nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size == 0U)) {
        return false;
    }
    if (heap->mode != NANOVEC_HEAP_MIN && heap->mode != NANOVEC_HEAP_MIN_MAX) {
        return false;
    }

    if (out != NULL) {
        *out = heap->elems[0];
    }

    size_t last = --heap->size;
    if (last > 0U) {
        if (heap->mode == NANOVEC_HEAP_MIN) {
            nanovec_heap_sift_down_min(heap->elems, last, heap->elems[last]);
        } else {
            heap->elems[0] = heap->elems[last];
            nanovec_minmax_trickle_down_min(heap->elems, last, 0U);
        }
    }
    return true;
}

bool nanovec_heap_pop_max(nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size == 0U)) {
        return false;
    }
    if (heap->mode != NANOVEC_HEAP_MAX && heap->mode != NANOVEC_HEAP_MIN_MAX) {
        return false;
    }

    if (heap->mode == NANOVEC_HEAP_MAX) {
        if (out != NULL) {
            *out = heap->elems[0];
        }
        size_t last = --heap->size;
        if (last > 0U) {
            nanovec_heap_sift_down_max(heap->elems, last, heap->elems[last]);
        }
        return true;
    }

    size_t max_idx = nanovec_minmax_max_index(heap);
    if (out != NULL) {
        *out = heap->elems[max_idx];
    }
    size_t last = --heap->size;
    if (max_idx < last) {
        heap->elems[max_idx] = heap->elems[last];
        nanovec_minmax_trickle_down_max(heap->elems, last, max_idx);
    }
    return true;
}

bool nanovec_heap_pop(nanovec_heap_t *heap, nanovec_heap_item_t *out) {
    if (NANOVEC_UNLIKELY(heap == NULL)) {
        return false;
    }
    if (heap->mode == NANOVEC_HEAP_MAX) {
        return nanovec_heap_pop_max(heap, out);
    }
    return nanovec_heap_pop_min(heap, out);
}

void nanovec_heap_sort_asc(nanovec_heap_t *heap) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || heap->size <= 1U)) {
        return;
    }

    size_t n = heap->size;
    if (heap->mode == NANOVEC_HEAP_MAX || heap->mode == NANOVEC_HEAP_MIN_MAX) {
        /* Repeatedly pop maximum to the back so elems[0..n-1] ends strictly ascending. */
        while (heap->size > 1U) {
            nanovec_heap_item_t item = {0ULL, 0.0f, 0U};
            if (!nanovec_heap_pop_max(heap, &item)) {
                break;
            }
            heap->elems[heap->size] = item;
        }
        heap->size = n;
        return;
    }

    /*
     * In MIN mode, in-place pop-to-back yields descending order; reverse once to ascending
     * with O(1) extra memory. Callers consume elems[0..size-1] sorted ascending.
     */
    while (heap->size > 1U) {
        nanovec_heap_item_t item = {0ULL, 0.0f, 0U};
        if (!nanovec_heap_pop_min(heap, &item)) {
            break;
        }
        heap->elems[heap->size] = item;
    }
    heap->size = n;
    for (size_t i = 0U, j = n - 1U; i < j; ++i, --j) {
        nanovec_heap_swap(&heap->elems[i], &heap->elems[j]);
    }
}

size_t nanovec_heap_to_results(
    nanovec_heap_t   *heap,
    nanovec_result_t *out_results,
    size_t            max_results
) {
    if (NANOVEC_UNLIKELY(heap == NULL || heap->elems == NULL || out_results == NULL || max_results == 0U)) {
        return 0U;
    }

    size_t n = heap->size;
    if (n == 0U) {
        return 0U;
    }

    size_t out_n = (n < max_results) ? n : max_results;

    if (heap->mode == NANOVEC_HEAP_MAX || heap->mode == NANOVEC_HEAP_MIN_MAX) {
        nanovec_heap_sort_asc(heap);
        for (size_t i = 0U; i < out_n; ++i) {
            out_results[i].id       = heap->elems[i].vid;
            out_results[i].distance = heap->elems[i].dist;
            out_results[i]._pad     = 0U;
        }
        heap->size = 0U;
        return out_n;
    }

    size_t written = 0U;
    for (size_t i = 0U; i < out_n; ++i) {
        nanovec_heap_item_t item = {0ULL, 0.0f, 0U};
        if (!nanovec_heap_pop_min(heap, &item)) {
            break;
        }
        out_results[i].id       = item.vid;
        out_results[i].distance = item.dist;
        out_results[i]._pad     = 0U;
        written++;
    }
    heap->size = 0U;
    return written;
}

size_t nanovec_heap_merge(nanovec_heap_t *dst, const nanovec_heap_t *src) {
    if (NANOVEC_UNLIKELY(
            dst == NULL || src == NULL || dst == src || dst->elems == NULL || src->elems == NULL
        )) {
        return 0U;
    }
    size_t accepted = 0U;
    for (size_t i = 0U; i < src->size; ++i) {
        const nanovec_heap_item_t *it = &src->elems[i];
        if (it->dist <= nanovec_heap_threshold(dst)) {
            if (nanovec_heap_push(dst, it->vid, it->dist)) {
                accepted++;
            }
        }
    }
    return accepted;
}
