/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "quant/adc.h"
#include "core/cpu.h"
#include "core/error.h"
#include "math/distance.h"
#include "math/prefetch.h"

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
