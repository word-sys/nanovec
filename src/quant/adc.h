/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_QUANT_ADC_H
#define NANOVEC_QUANT_ADC_H

#include "core/alloc.h"
#include "core/types.h"
#include "quant/pq.h"

typedef struct nanovec_adc_table {
    uint16_t         m;
    uint16_t         ksub;
    nanovec_metric_t metric;
    bool             owns_lut;
    float           *lut; /* [m][256], 64-byte aligned */
} nanovec_adc_table_t;

static inline const float *nanovec_adc_lut_row(
    const nanovec_adc_table_t *tbl,
    uint32_t                   subspace
) {
    return tbl->lut + ((size_t)subspace * NANOVEC_PQ_K);
}

nanovec_status_t nanovec_adc_table_init(nanovec_adc_table_t *tbl, uint16_t m);
nanovec_status_t nanovec_adc_table_init_arena(
    nanovec_adc_table_t *tbl,
    uint16_t             m,
    nanovec_arena_t     *arena
);
void nanovec_adc_table_destroy(nanovec_adc_table_t *tbl);

nanovec_status_t nanovec_adc_table_compute(
    nanovec_adc_table_t *tbl,
    const nanovec_pq_t  *pq,
    const float         *query,
    nanovec_metric_t     metric
);

nanovec_status_t nanovec_adc_build(
    const nanovec_pq_t  *pq,
    const float         *query,
    nanovec_metric_t     metric,
    nanovec_adc_table_t *out_tbl
);

float nanovec_adc_distance_scalar(const nanovec_adc_table_t *tbl, const uint8_t *pq_code);
float nanovec_adc_distance(const nanovec_adc_table_t *tbl, const uint8_t *pq_code);
void  nanovec_adc_distance_batch(
    const nanovec_adc_table_t *tbl,
    const uint8_t             *pq_codes,
    size_t                     count,
    float                     *out_dists
);

typedef const float *(*nanovec_vec_fetch_fn_t)(
    uint32_t slot,
    void    *user_ctx,
    float   *scratch_buf
);

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
);

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
);

#endif /* NANOVEC_QUANT_ADC_H */
