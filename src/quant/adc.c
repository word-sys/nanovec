/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "quant/adc.h"
#include "core/cpu.h"
#include "core/error.h"
#include "math/distance.h"
#include "math/prefetch.h"
#include "quant/bq.h"

#include <math.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

nanovec_status_t nanovec_adc_table_init(nanovec_adc_table_t *tbl, uint16_t m) {
    if (NANOVEC_UNLIKELY(tbl == NULL || m == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "adc_table_init requires non-null tbl and m > 0"
        );
    }

    memset(tbl, 0, sizeof(*tbl));
    float *lut = (float *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)m * NANOVEC_PQ_K * sizeof(float)
    );
    if (NANOVEC_UNLIKELY(lut == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    tbl->m        = m;
    tbl->ksub     = (uint16_t)NANOVEC_PQ_K;
    tbl->metric   = NANOVEC_METRIC_L2;
    tbl->owns_lut = true;
    tbl->lut      = lut;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_adc_table_init_arena(
    nanovec_adc_table_t *tbl,
    uint16_t             m,
    nanovec_arena_t     *arena
) {
    if (NANOVEC_UNLIKELY(tbl == NULL || m == 0 || arena == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "adc_table_init_arena requires non-null tbl, arena, and m > 0"
        );
    }

    memset(tbl, 0, sizeof(*tbl));
    float *lut = (float *)nanovec_arena_alloc(
        arena,
        (size_t)m * NANOVEC_PQ_K * sizeof(float),
        NANOVEC_CACHE_LINE
    );
    if (NANOVEC_UNLIKELY(lut == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    tbl->m        = m;
    tbl->ksub     = (uint16_t)NANOVEC_PQ_K;
    tbl->metric   = NANOVEC_METRIC_L2;
    tbl->owns_lut = false;
    tbl->lut      = lut;
    return NANOVEC_SUCCESS;
}

void nanovec_adc_table_destroy(nanovec_adc_table_t *tbl) {
    if (tbl == NULL) {
        return;
    }
    if (tbl->owns_lut) {
        nanovec_aligned_free(tbl->lut);
    }
    memset(tbl, 0, sizeof(*tbl));
}

nanovec_status_t nanovec_adc_table_compute(
    nanovec_adc_table_t *tbl,
    const nanovec_pq_t  *pq,
    const float         *query,
    nanovec_metric_t     metric
) {
    if (NANOVEC_UNLIKELY(
            tbl == NULL || tbl->lut == NULL || pq == NULL || !pq->trained ||
            query == NULL || tbl->m != pq->m
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "adc_table_compute requires initialized tbl matching trained pq and non-null query"
        );
    }

    uint32_t m       = pq->m;
    uint32_t sub_dim = pq->sub_dim;
    uint32_t ksub    = pq->ksub;

    if (metric == NANOVEC_METRIC_L2 || metric == NANOVEC_METRIC_DOT) {
        nanovec_dist_f32_fn_t dist_fn =
            (metric == NANOVEC_METRIC_L2) ? nanovec_dist_l2_fn : nanovec_dist_dot_fn;

        for (uint32_t s = 0; s < m; ++s) {
            const float *q_sub = query + (size_t)s * sub_dim;
            const float *cents = pq->centroids + ((size_t)s * ksub * sub_dim);
            float       *row   = tbl->lut + ((size_t)s * ksub);

            for (uint32_t k = 0; k < ksub; ++k) {
                if (k + 1U < ksub) {
                    nanovec_prefetch_l1(cents + (size_t)(k + 1U) * sub_dim);
                }
                row[k] = dist_fn(q_sub, cents + (size_t)k * sub_dim, sub_dim);
            }
        }
        tbl->metric = metric;
        return NANOVEC_SUCCESS;
    }

    if (metric == NANOVEC_METRIC_COSINE) {
        double norm_sq = 0.0;
        for (uint32_t d = 0; d < pq->dim; ++d) {
            norm_sq += (double)query[d] * (double)query[d];
        }
        float bias = 1.0f / (float)m;
        if (norm_sq <= 0.0) {
            size_t total = (size_t)m * ksub;
            for (size_t i = 0; i < total; ++i) {
                tbl->lut[i] = bias;
            }
            tbl->metric = metric;
            return NANOVEC_SUCCESS;
        }

        float inv_norm = (float)(1.0 / sqrt(norm_sq));
        nanovec_dist_f32_fn_t dot_fn = nanovec_dist_dot_fn;

        for (uint32_t s = 0; s < m; ++s) {
            const float *q_sub = query + (size_t)s * sub_dim;
            const float *cents = pq->centroids + ((size_t)s * ksub * sub_dim);
            float       *row   = tbl->lut + ((size_t)s * ksub);

            for (uint32_t k = 0; k < ksub; ++k) {
                if (k + 1U < ksub) {
                    nanovec_prefetch_l1(cents + (size_t)(k + 1U) * sub_dim);
                }
                row[k] = bias + dot_fn(q_sub, cents + (size_t)k * sub_dim, sub_dim) * inv_norm;
            }
        }
        tbl->metric = metric;
        return NANOVEC_SUCCESS;
    }

    return (nanovec_status_t)nanovec_set_error(
        NANOVEC_ERR_UNSUPPORTED,
        "adc_table_compute does not support metric %d",
        (int)metric
    );
}

nanovec_status_t nanovec_adc_build(
    const nanovec_pq_t  *pq,
    const float         *query,
    nanovec_metric_t     metric,
    nanovec_adc_table_t *out_tbl
) {
    if (NANOVEC_UNLIKELY(pq == NULL || out_tbl == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "adc_build requires non-null pq and out_tbl"
        );
    }

    nanovec_status_t st = nanovec_adc_table_init(out_tbl, pq->m);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    st = nanovec_adc_table_compute(out_tbl, pq, query, metric);
    if (st != NANOVEC_SUCCESS) {
        nanovec_adc_table_destroy(out_tbl);
        return st;
    }
    return NANOVEC_SUCCESS;
}

float nanovec_adc_distance_scalar(const nanovec_adc_table_t *tbl, const uint8_t *pq_code) {
    if (NANOVEC_UNLIKELY(tbl == NULL || tbl->lut == NULL || pq_code == NULL)) {
        return 0.0f;
    }

    const float *lut = tbl->lut;
    uint32_t m = tbl->m;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    uint32_t s = 0;
    uint32_t limit = m & ~3U;
    for (; s < limit; s += 4U) {
        const float *base = lut + (size_t)s * NANOVEC_PQ_K;
        s0 += base[pq_code[s]];
        s1 += base[NANOVEC_PQ_K + pq_code[s + 1]];
        s2 += base[2U * NANOVEC_PQ_K + pq_code[s + 2]];
        s3 += base[3U * NANOVEC_PQ_K + pq_code[s + 3]];
    }

    float tail = 0.0f;
    for (; s < m; ++s) {
        tail += lut[(size_t)s * NANOVEC_PQ_K + pq_code[s]];
    }
    return ((s0 + s1) + (s2 + s3)) + tail;
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static float nanovec_adc_distance_avx2(const nanovec_adc_table_t *tbl, const uint8_t *pq_code) {
    const float *lut = tbl->lut;
    uint32_t m = tbl->m;

    const __m256i row_off = _mm256_setr_epi32(
        0, 256, 512, 768, 1024, 1280, 1536, 1792
    );
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    uint32_t s = 0;
    uint32_t limit16 = m & ~15U;
    for (; s < limit16; s += 16U) {
        uint64_t raw0, raw1;
        memcpy(&raw0, pq_code + s, 8);
        memcpy(&raw1, pq_code + s + 8, 8);

        __m256i idx0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)raw0)), row_off);
        __m256i idx1 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)raw1)), row_off);

        __m256 v0 = _mm256_i32gather_ps(lut + (size_t)s * NANOVEC_PQ_K, idx0, 4);
        __m256 v1 = _mm256_i32gather_ps(lut + (size_t)(s + 8U) * NANOVEC_PQ_K, idx1, 4);

        acc0 = _mm256_add_ps(acc0, v0);
        acc1 = _mm256_add_ps(acc1, v1);
    }

    acc0 = _mm256_add_ps(acc0, acc1);

    if (s + 8U <= m) {
        uint64_t raw;
        memcpy(&raw, pq_code + s, 8);
        __m256i idx = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)raw)), row_off);
        acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(lut + (size_t)s * NANOVEC_PQ_K, idx, 4));
        s += 8U;
    }

    __m256 t = _mm256_hadd_ps(acc0, acc0);
    t = _mm256_hadd_ps(t, t);
    __m128 lo = _mm256_castps256_ps128(t);
    __m128 hi = _mm256_extractf128_ps(t, 1);
    float total = _mm_cvtss_f32(_mm_add_ss(lo, hi));

    for (; s < m; ++s) {
        total += lut[(size_t)s * NANOVEC_PQ_K + pq_code[s]];
    }
    return total;
}
#endif

float nanovec_adc_distance(const nanovec_adc_table_t *tbl, const uint8_t *pq_code) {
    if (NANOVEC_UNLIKELY(tbl == NULL || tbl->lut == NULL || pq_code == NULL)) {
        return 0.0f;
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (nanovec_cpu_get_caps()->avx2) {
        return nanovec_adc_distance_avx2(tbl, pq_code);
    }
#endif
    return nanovec_adc_distance_scalar(tbl, pq_code);
}

void nanovec_adc_distance_batch(
    const nanovec_adc_table_t *tbl,
    const uint8_t             *pq_codes,
    size_t                     count,
    float                     *out_dists
) {
    if (NANOVEC_UNLIKELY(
            tbl == NULL || tbl->lut == NULL || (count > 0 && (pq_codes == NULL || out_dists == NULL))
        )) {
        return;
    }

    size_t stride = tbl->m;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    bool use_avx2 = nanovec_cpu_get_caps()->avx2 != 0;
#endif
    for (size_t i = 0; i < count; ++i) {
        if (i + 2U < count) {
            nanovec_prefetch_l1(pq_codes + (i + 2U) * stride);
        }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if (use_avx2) {
            out_dists[i] = nanovec_adc_distance_avx2(tbl, pq_codes + i * stride);
            continue;
        }
#endif
        out_dists[i] = nanovec_adc_distance_scalar(tbl, pq_codes + i * stride);
    }
}

#define NANOVEC_RERANK_DEFAULT_FACTOR 10U

typedef struct nanovec_cand {
    float    dist;
    uint32_t slot;
} nanovec_cand_t;

static inline bool nanovec_cand_worse(nanovec_cand_t a, nanovec_cand_t b) {
    return a.dist > b.dist || (a.dist == b.dist && a.slot > b.slot);
}

static void nanovec_maxheap_sift_down(nanovec_cand_t *h, uint32_t n, uint32_t idx) {
    for (;;) {
        uint32_t left  = (idx << 1) + 1U;
        uint32_t right = left + 1U;
        uint32_t worst = idx;

        if (left < n && nanovec_cand_worse(h[left], h[worst])) {
            worst = left;
        }
        if (right < n && nanovec_cand_worse(h[right], h[worst])) {
            worst = right;
        }
        if (worst == idx) {
            break;
        }
        nanovec_cand_t tmp = h[idx];
        h[idx]   = h[worst];
        h[worst] = tmp;
        idx = worst;
    }
}

static void nanovec_maxheap_push(
    nanovec_cand_t *h,
    uint32_t       *size,
    uint32_t        cap,
    float           dist,
    uint32_t        slot
) {
    nanovec_cand_t item = { dist, slot };
    if (*size < cap) {
        uint32_t idx = (*size)++;
        h[idx] = item;
        while (idx > 0U) {
            uint32_t parent = (idx - 1U) >> 1;
            if (!nanovec_cand_worse(h[idx], h[parent])) {
                break;
            }
            nanovec_cand_t tmp = h[parent];
            h[parent] = h[idx];
            h[idx]    = tmp;
            idx = parent;
        }
    } else if (nanovec_cand_worse(h[0], item)) {
        h[0] = item;
        nanovec_maxheap_sift_down(h, cap, 0U);
    }
}

static void nanovec_maxheap_sort_asc(nanovec_cand_t *h, uint32_t n) {
    for (uint32_t end = n; end > 1U; --end) {
        nanovec_cand_t tmp = h[0];
        h[0]       = h[end - 1U];
        h[end - 1U] = tmp;
        nanovec_maxheap_sift_down(h, end - 1U, 0U);
    }
}

static uint32_t nanovec_resolve_coarse_n(size_t count, uint32_t top_k, uint32_t rerank_factor) {
    uint32_t factor = rerank_factor > 0U ? rerank_factor : NANOVEC_RERANK_DEFAULT_FACTOR;
    if (top_k >= count) {
        return (uint32_t)count;
    }
    if ((size_t)top_k > count / (size_t)factor) {
        return (uint32_t)count;
    }
    uint32_t n = top_k * factor;
    return n < top_k ? top_k : n;
}

static nanovec_status_t nanovec_rerank_fine_pass(
    const float            *query,
    uint32_t                dim,
    const nanovec_cand_t   *coarse,
    uint32_t                coarse_n,
    const float            *fp32_vectors,
    const uint64_t         *vids,
    nanovec_dist_f32_fn_t   dist_fn,
    uint32_t                k,
    nanovec_vec_fetch_fn_t  fetch_fn,
    void                   *fetch_ctx,
    nanovec_result_t       *out_results,
    size_t                 *out_count
) {
    nanovec_cand_t *fine = (nanovec_cand_t *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)k * sizeof(nanovec_cand_t)
    );
    if (NANOVEC_UNLIKELY(fine == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    float *scratch = NULL;
    if (fp32_vectors == NULL) {
        scratch = (float *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, (size_t)dim * sizeof(float));
        if (NANOVEC_UNLIKELY(scratch == NULL)) {
            nanovec_aligned_free(fine);
            return NANOVEC_ERR_OOM;
        }
    }

    uint32_t fine_size = 0U;
    for (uint32_t i = 0; i < coarse_n; ++i) {
        uint32_t slot = coarse[i].slot;
        const float *vec = NULL;
        if (fp32_vectors != NULL) {
            if (i + 1U < coarse_n) {
                nanovec_prefetch_l1(fp32_vectors + (size_t)coarse[i + 1U].slot * dim);
            }
            vec = fp32_vectors + (size_t)slot * dim;
        } else {
            vec = fetch_fn(slot, fetch_ctx, scratch);
            if (NANOVEC_UNLIKELY(vec == NULL)) {
                continue;
            }
        }
        float exact_d = dist_fn(query, vec, dim);
        nanovec_maxheap_push(fine, &fine_size, k, exact_d, slot);
    }

    nanovec_maxheap_sort_asc(fine, fine_size);
    for (uint32_t i = 0; i < fine_size; ++i) {
        out_results[i].id       = vids != NULL ? vids[fine[i].slot] : (uint64_t)fine[i].slot;
        out_results[i].distance = fine[i].dist;
        out_results[i]._pad     = 0U;
    }
    *out_count = fine_size;

    nanovec_aligned_free(scratch);
    nanovec_aligned_free(fine);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_rerank_bq(
    const float            *query,
    uint32_t                dim,
    const uint64_t         *bq_codes,
    const float            *fp32_vectors,
    const uint64_t         *vids,
    size_t                  count,
    nanovec_metric_t        metric,
    uint32_t                top_k,
    uint32_t                rerank_factor,
    nanovec_vec_fetch_fn_t  fetch_fn,
    void                   *fetch_ctx,
    nanovec_result_t       *out_results,
    size_t                 *out_count
) {
    if (NANOVEC_UNLIKELY(out_count == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "out_count is null");
    }
    *out_count = 0U;

    if (count == 0U || top_k == 0U) {
        return NANOVEC_SUCCESS;
    }
    if (NANOVEC_UNLIKELY(
            query == NULL || dim == 0U || bq_codes == NULL || out_results == NULL ||
            (fp32_vectors == NULL && fetch_fn == NULL)
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "rerank_bq requires valid query, dim, bq_codes, out_results, and vector source"
        );
    }

    nanovec_dist_f32_fn_t dist_fn = nanovec_get_dist_f32_fn(metric);
    if (NANOVEC_UNLIKELY(dist_fn == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_UNSUPPORTED,
            "rerank_bq requires an FP32 metric (L2, COSINE, or DOT)"
        );
    }

    uint32_t k        = (uint32_t)(top_k < count ? top_k : count);
    uint32_t coarse_n = nanovec_resolve_coarse_n(count, k, rerank_factor);
    uint32_t words    = nanovec_bq_num_words(dim);

    uint64_t *q_bq = (uint64_t *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)words * sizeof(uint64_t)
    );
    nanovec_cand_t *coarse = (nanovec_cand_t *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)coarse_n * sizeof(nanovec_cand_t)
    );
    if (NANOVEC_UNLIKELY(!q_bq || !coarse)) {
        nanovec_aligned_free(q_bq);
        nanovec_aligned_free(coarse);
        return NANOVEC_ERR_OOM;
    }

    nanovec_status_t st = nanovec_bq_encode(query, dim, q_bq);
    if (NANOVEC_UNLIKELY(st != NANOVEC_SUCCESS)) {
        nanovec_aligned_free(q_bq);
        nanovec_aligned_free(coarse);
        return st;
    }

    uint32_t coarse_size = 0U;
    for (size_t i = 0; i < count; ++i) {
        if (i + 2U < count) {
            nanovec_prefetch_l1(bq_codes + (i + 2U) * words);
        }
        uint32_t h = nanovec_bq_distance(q_bq, bq_codes + i * words, words);
        nanovec_maxheap_push(coarse, &coarse_size, coarse_n, (float)h, (uint32_t)i);
    }

    nanovec_aligned_free(q_bq);

    st = nanovec_rerank_fine_pass(
        query, dim, coarse, coarse_size, fp32_vectors, vids,
        dist_fn, k, fetch_fn, fetch_ctx, out_results, out_count
    );
    nanovec_aligned_free(coarse);
    return st;
}

nanovec_status_t nanovec_rerank_pq(
    const nanovec_pq_t     *pq,
    const float            *query,
    const uint8_t          *pq_codes,
    const float            *fp32_vectors,
    const uint64_t         *vids,
    size_t                  count,
    nanovec_metric_t        metric,
    uint32_t                top_k,
    uint32_t                rerank_factor,
    nanovec_vec_fetch_fn_t  fetch_fn,
    void                   *fetch_ctx,
    nanovec_result_t       *out_results,
    size_t                 *out_count
) {
    if (NANOVEC_UNLIKELY(out_count == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "out_count is null");
    }
    *out_count = 0U;

    if (count == 0U || top_k == 0U) {
        return NANOVEC_SUCCESS;
    }
    if (NANOVEC_UNLIKELY(
            pq == NULL || !pq->trained || query == NULL || pq_codes == NULL ||
            out_results == NULL || (fp32_vectors == NULL && fetch_fn == NULL)
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "rerank_pq requires trained pq, query, pq_codes, out_results, and vector source"
        );
    }

    nanovec_dist_f32_fn_t dist_fn = nanovec_get_dist_f32_fn(metric);
    if (NANOVEC_UNLIKELY(dist_fn == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_UNSUPPORTED,
            "rerank_pq requires an FP32 metric (L2, COSINE, or DOT)"
        );
    }

    nanovec_adc_table_t tbl;
    nanovec_status_t st = nanovec_adc_build(pq, query, metric, &tbl);
    if (NANOVEC_UNLIKELY(st != NANOVEC_SUCCESS)) {
        return st;
    }

    uint32_t k        = (uint32_t)(top_k < count ? top_k : count);
    uint32_t coarse_n = nanovec_resolve_coarse_n(count, k, rerank_factor);
    size_t   m        = pq->m;

    nanovec_cand_t *coarse = (nanovec_cand_t *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)coarse_n * sizeof(nanovec_cand_t)
    );
    if (NANOVEC_UNLIKELY(coarse == NULL)) {
        nanovec_adc_table_destroy(&tbl);
        return NANOVEC_ERR_OOM;
    }

    uint32_t coarse_size = 0U;
    for (size_t i = 0; i < count; ++i) {
        if (i + 2U < count) {
            nanovec_prefetch_l1(pq_codes + (i + 2U) * m);
        }
        float d = nanovec_adc_distance(&tbl, pq_codes + i * m);
        nanovec_maxheap_push(coarse, &coarse_size, coarse_n, d, (uint32_t)i);
    }

    nanovec_adc_table_destroy(&tbl);

    st = nanovec_rerank_fine_pass(
        query, pq->dim, coarse, coarse_size, fp32_vectors, vids,
        dist_fn, k, fetch_fn, fetch_ctx, out_results, out_count
    );
    nanovec_aligned_free(coarse);
    return st;
}
