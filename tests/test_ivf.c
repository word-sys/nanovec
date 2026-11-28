/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "index/heap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int cmp_item_asc(const void *pa, const void *pb) {
    const nanovec_heap_item_t *a = (const nanovec_heap_item_t *)pa;
    const nanovec_heap_item_t *b = (const nanovec_heap_item_t *)pb;
    if (a->dist < b->dist) return -1;
    if (a->dist > b->dist) return 1;
    if (a->vid < b->vid) return -1;
    if (a->vid > b->vid) return 1;
    return 0;
}

static size_t node_depth(size_t idx) {
    size_t d = 0U;
    size_t k = idx + 1U;
    while (k > 1U) {
        k >>= 1U;
        d++;
    }
    return d;
}

/* Exhaustively verifies the Atkinson Min-Max heap invariant over every node's subtree. */
static void verify_minmax_invariant(const nanovec_heap_t *h) {
    CHECK(h->mode == NANOVEC_HEAP_MIN_MAX);
    for (size_t i = 0U; i < h->size; ++i) {
        bool is_min = ((node_depth(i) & 1U) == 0U);
        for (size_t j = i + 1U; j < h->size; ++j) {
            size_t anc = j;
            while (anc > i) {
                anc = (anc - 1U) >> 1U;
                if (anc == i) {
                    const nanovec_heap_item_t *ej = &h->elems[j];
                    const nanovec_heap_item_t *ei = &h->elems[i];
                    if (is_min) {
                        CHECK(!nanovec_heap_item_lt(*ej, *ei));
                    } else {
                        CHECK(!nanovec_heap_item_lt(*ei, *ej));
                    }
                    break;
                }
            }
        }
    }
}

static void verify_max_invariant(const nanovec_heap_t *h) {
    CHECK(h->mode == NANOVEC_HEAP_MAX);
    for (size_t i = 1U; i < h->size; ++i) {
        size_t p = (i - 1U) >> 1U;
        CHECK(!nanovec_heap_item_lt(h->elems[p], h->elems[i]));
    }
}

static void verify_min_invariant(const nanovec_heap_t *h) {
    CHECK(h->mode == NANOVEC_HEAP_MIN);
    for (size_t i = 1U; i < h->size; ++i) {
        size_t p = (i - 1U) >> 1U;
        CHECK(!nanovec_heap_item_lt(h->elems[i], h->elems[p]));
    }
}

static void test_heap_alignment_and_modes(void) {
    nanovec_heap_t max_h, min_h, mm_h;
    CHECK(nanovec_heap_init(&max_h, 32U, NANOVEC_HEAP_MAX) == NANOVEC_SUCCESS);
    CHECK(nanovec_heap_init(&min_h, 32U, NANOVEC_HEAP_MIN) == NANOVEC_SUCCESS);
    CHECK(nanovec_heap_init(&mm_h, 32U, NANOVEC_HEAP_MIN_MAX) == NANOVEC_SUCCESS);

    CHECK(((uintptr_t)max_h.elems & 63U) == 0U);
    CHECK(((uintptr_t)min_h.elems & 63U) == 0U);
    CHECK(((uintptr_t)mm_h.elems & 63U) == 0U);
    CHECK(nanovec_heap_empty(&max_h));
    CHECK(!nanovec_heap_full(&max_h));
    CHECK(isinf(nanovec_heap_threshold(&max_h)) && nanovec_heap_threshold(&max_h) > 0.0f);

    /* Arena-backed heap allocation works without separate free. */
    nanovec_arena_t arena;
    CHECK(nanovec_arena_init(&arena, 4096U) == NANOVEC_SUCCESS);
    nanovec_heap_t arena_h;
    CHECK(nanovec_heap_init_arena(&arena_h, &arena, 16U, NANOVEC_HEAP_MAX) == NANOVEC_SUCCESS);
    CHECK(((uintptr_t)arena_h.elems & 63U) == 0U);
    CHECK(!arena_h.owns_mem);
    CHECK(nanovec_heap_push(&arena_h, 11ULL, 2.5f));
    CHECK(nanovec_heap_push(&arena_h, 22ULL, 1.5f));
    nanovec_result_t res[2];
    CHECK(nanovec_heap_to_results(&arena_h, res, 2U) == 2U);
    CHECK(res[0].id == 22ULL && res[0].distance == 1.5f);
    CHECK(res[1].id == 11ULL && res[1].distance == 2.5f);
    nanovec_heap_destroy(&arena_h);
    nanovec_arena_destroy(&arena);

    /* Stack/external buffer-backed heap works cleanly. */
    NANOVEC_ALIGNAS(64) nanovec_heap_item_t stack_buf[8];
    nanovec_heap_t ext_h;
    CHECK(nanovec_heap_init_buffer(&ext_h, stack_buf, 8U, NANOVEC_HEAP_MIN_MAX) == NANOVEC_SUCCESS);
    CHECK(!ext_h.owns_mem);
    CHECK(nanovec_heap_push(&ext_h, 1ULL, 9.0f));
    CHECK(nanovec_heap_push(&ext_h, 2ULL, 3.0f));
    nanovec_heap_item_t item;
    CHECK(nanovec_heap_peek_min(&ext_h, &item) && item.vid == 2ULL);
    CHECK(nanovec_heap_peek_max(&ext_h, &item) && item.vid == 1ULL);
    nanovec_heap_destroy(&ext_h);

    /* NaN rejection & invalid arg checks. */
    CHECK(!nanovec_heap_push(&max_h, 1ULL, NAN));
    CHECK(nanovec_heap_empty(&max_h));
    nanovec_heap_t bad_h;
    CHECK(nanovec_heap_init(NULL, 8U, NANOVEC_HEAP_MAX) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_heap_init(&bad_h, 8U, (nanovec_heap_mode_t)99) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_heap_init_buffer(&bad_h, NULL, 8U, NANOVEC_HEAP_MAX) == NANOVEC_ERR_INVALID_ARG);

    nanovec_heap_destroy(&max_h);
    nanovec_heap_destroy(&min_h);
    nanovec_heap_destroy(&mm_h);
}

static void test_topk_and_minmax_equivalence(void) {
    const size_t N = 5000U;
    const size_t K = 17U;

    nanovec_heap_item_t *all = (nanovec_heap_item_t *)malloc(N * sizeof(nanovec_heap_item_t));
    CHECK(all != NULL);

    nanovec_heap_t h_max, h_mm;
    CHECK(nanovec_heap_init(&h_max, K, NANOVEC_HEAP_MAX) == NANOVEC_SUCCESS);
    CHECK(nanovec_heap_init(&h_mm, K, NANOVEC_HEAP_MIN_MAX) == NANOVEC_SUCCESS);

    uint64_t rng = 0x123456789ABCDEF0ULL;
    for (size_t i = 0U; i < N; ++i) {
        /* Mix continuous floats with exact ties to stress tie-breaking by VID. */
        uint64_t r = rng_next(&rng);
        float    d = (i % 13U == 0U)
            ? (float)(r % 7ULL) * 0.25f
            : (float)(r & 0xFFFFFFULL) / (float)0x100000ULL;
        nanovec_vid_t vid = r ^ (uint64_t)i;
        all[i].vid  = vid;
        all[i].dist = d;
        all[i]._pad = 0U;

        if (d <= nanovec_heap_threshold(&h_max)) {
            (void)nanovec_heap_push(&h_max, vid, d);
        }
        if (d <= nanovec_heap_threshold(&h_mm)) {
            (void)nanovec_heap_push(&h_mm, vid, d);
        }

        if ((i % 250U) == 0U || i == N - 1U) {
            verify_max_invariant(&h_max);
            verify_minmax_invariant(&h_mm);
        }
    }

    qsort(all, N, sizeof(all[0]), cmp_item_asc);

    nanovec_result_t res_max[17], res_mm[17];
    CHECK(nanovec_heap_to_results(&h_max, res_max, K) == K);
    CHECK(nanovec_heap_to_results(&h_mm, res_mm, K) == K);
    for (size_t i = 0U; i < K; ++i) {
        CHECK(res_max[i].id == all[i].vid && res_max[i].distance == all[i].dist);
        CHECK(res_mm[i].id == all[i].vid && res_mm[i].distance == all[i].dist);
    }

    nanovec_heap_destroy(&h_max);
    nanovec_heap_destroy(&h_mm);
    free(all);
}

static void test_minmax_fuzz_and_duplicates(void) {
    /* Heavy interleaved push / pop_min / pop_max / bounded-replace with duplicate items. */
    const size_t CAP = 31U;
    nanovec_heap_t h_mm;
    CHECK(nanovec_heap_init(&h_mm, CAP, NANOVEC_HEAP_MIN_MAX) == NANOVEC_SUCCESS);

    uint64_t rng = 0xCAFEBABEDEADBEEFULL;
    for (int step = 0; step < 3000; ++step) {
        uint64_t r = rng_next(&rng);
        int      op = (int)(r % 5ULL);
        if (op <= 2) {
            /* Force frequent duplicate (dist, vid) pairs as well as unique values. */
            nanovec_vid_t  vid  = (r >> 4) % 12ULL;
            nanovec_dist_t dist = (float)((r >> 12) % 9ULL) * 0.5f;
            (void)nanovec_heap_push(&h_mm, vid, dist);
        } else if (op == 3 && !nanovec_heap_empty(&h_mm)) {
            nanovec_heap_item_t pmin, out;
            CHECK(nanovec_heap_peek_min(&h_mm, &pmin));
            CHECK(nanovec_heap_pop_min(&h_mm, &out));
            CHECK(pmin.vid == out.vid && pmin.dist == out.dist);
        } else if (op == 4 && !nanovec_heap_empty(&h_mm)) {
            nanovec_heap_item_t pmax, out;
            CHECK(nanovec_heap_peek_max(&h_mm, &pmax));
            CHECK(nanovec_heap_pop_max(&h_mm, &out));
            CHECK(pmax.vid == out.vid && pmax.dist == out.dist);
        }
        verify_minmax_invariant(&h_mm);
    }

    /* Check K=1, K=2, K=3 boundary capacities in MIN_MAX mode. */
    for (size_t k = 1U; k <= 3U; ++k) {
        nanovec_heap_t hk;
        CHECK(nanovec_heap_init(&hk, k, NANOVEC_HEAP_MIN_MAX) == NANOVEC_SUCCESS);
        const float vals[6] = {5.0f, 1.0f, 4.0f, 0.5f, 3.0f, 0.25f};
        for (size_t i = 0U; i < 6U; ++i) {
            (void)nanovec_heap_push(&hk, (nanovec_vid_t)(i + 1U), vals[i]);
            verify_minmax_invariant(&hk);
        }
        nanovec_heap_item_t item;
        CHECK(nanovec_heap_peek_min(&hk, &item) && item.dist == 0.25f);
        nanovec_result_t out[3];
        CHECK(nanovec_heap_to_results(&hk, out, k) == k);
        CHECK(out[0].distance == 0.25f);
        if (k >= 2U) CHECK(out[1].distance == 0.5f);
        if (k >= 3U) CHECK(out[2].distance == 1.0f);
        nanovec_heap_destroy(&hk);
    }

    nanovec_heap_destroy(&h_mm);
}

static void test_min_heap_and_merge(void) {
    nanovec_heap_t hmin;
    CHECK(nanovec_heap_init(&hmin, 200U, NANOVEC_HEAP_MIN) == NANOVEC_SUCCESS);

    uint64_t rng = 0x9988776655443322ULL;
    for (size_t i = 0U; i < 250U; ++i) {
        nanovec_dist_t d  = (float)(rng_next(&rng) % 10000ULL) * 0.01f;
        bool           ok = nanovec_heap_push(&hmin, (nanovec_vid_t)i, d);
        CHECK(ok == (i < 200U));
        if ((i % 25U) == 0U) {
            verify_min_invariant(&hmin);
        }
    }

    nanovec_heap_sort_asc(&hmin);
    CHECK(nanovec_heap_size(&hmin) == 200U);
    for (size_t i = 1U; i < 200U; ++i) {
        CHECK(!nanovec_heap_item_lt(hmin.elems[i], hmin.elems[i - 1U]));
    }
    nanovec_heap_destroy(&hmin);

    /* Simulate Phase 48 multi-thread partition merge across 4 shards into top-10 master heap. */
    const size_t   K = 10U;
    nanovec_heap_t master;
    nanovec_heap_t shards[4];
    CHECK(nanovec_heap_init(&master, K, NANOVEC_HEAP_MAX) == NANOVEC_SUCCESS);
    for (int s = 0; s < 4; ++s) {
        CHECK(nanovec_heap_init(&shards[s], K, NANOVEC_HEAP_MAX) == NANOVEC_SUCCESS);
    }

    for (uint64_t id = 0ULL; id < 400ULL; ++id) {
        float dist = (float)((id * 173ULL + 31ULL) % 400ULL) + 0.5f;
        (void)nanovec_heap_push(&shards[id % 4ULL], id, dist);
    }
    for (int s = 0; s < 4; ++s) {
        (void)nanovec_heap_merge(&master, &shards[s]);
        nanovec_heap_destroy(&shards[s]);
    }

    nanovec_result_t res[10];
    CHECK(nanovec_heap_to_results(&master, res, K) == K);
    for (size_t i = 0U; i < K; ++i) {
        CHECK(res[i].distance == (float)i + 0.5f);
    }
    nanovec_heap_destroy(&master);
}

int main(void) {
    test_heap_alignment_and_modes();
    test_topk_and_minmax_equivalence();
    test_minmax_fuzz_and_duplicates();
    test_min_heap_and_merge();
    (void)printf("test_ivf: Phase 46 heap tests PASSED\n");
    return 0;
}
