/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "quant/adc.h"
#include "core/error.h"
#include "math/distance.h"
#include "math/prefetch.h"

#include <math.h>
#include <string.h>

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
