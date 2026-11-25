/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "core/alloc.h"
#include "core/cpu.h"
#include "math/distance.h"
#include "quant/adc.h"
#include "quant/bq.h"
#include "quant/kmeans.h"
#include "quant/pq.h"
#include "quant/sq8.h"

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

#define TEST_N        1200U
#define TEST_DIM      384U
#define TEST_PQ_M     48U
#define TEST_CLUSTERS 16U

static uint64_t g_rng = 0x33333333CAFEBABEULL;

static uint64_t rng_u64(void) {
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_u01(void) {
    return ((double)(rng_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static float rng_normal(void) {
    double u1 = rng_u01();
    double u2 = rng_u01();
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2));
}

static void normalize_vec(float *v, uint32_t dim) {
    double s = 0.0;
    for (uint32_t d = 0; d < dim; ++d) {
        s += (double)v[d] * (double)v[d];
    }
    if (s > 0.0) {
        float inv = (float)(1.0 / sqrt(s));
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] *= inv;
        }
    }
}

static void generate_clustered_embeddings(float *out, size_t n, uint32_t dim) {
    float *centers = (float *)nanovec_aligned_alloc(64, (size_t)TEST_CLUSTERS * dim * sizeof(float));
    CHECK(centers != NULL);

    for (uint32_t c = 0; c < TEST_CLUSTERS; ++c) {
        float *ctr = centers + (size_t)c * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            ctr[d] = rng_normal();
        }
        normalize_vec(ctr, dim);
    }

    for (size_t i = 0; i < n; ++i) {
        const float *ctr = centers + (size_t)(i % TEST_CLUSTERS) * dim;
        float *v = out + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] = ctr[d] + 0.04f * rng_normal();
        }
        normalize_vec(v, dim);
    }

    nanovec_aligned_free(centers);
}

static void compute_distortion(
    const float *orig,
    const float *recon,
    size_t       n,
    uint32_t     dim,
    double      *out_mse,
    double      *out_cos_distort
) {
    double sum_sq  = 0.0;
    double sum_cos = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float *a = orig + i * dim;
        const float *b = recon + i * dim;
        sum_sq  += (double)nanovec_dist_l2_fn(a, b, dim);
        sum_cos += (double)nanovec_dist_cosine_fn(a, b, dim);
    }
    *out_mse         = sum_sq / (double)(n * dim);
    *out_cos_distort = sum_cos / (double)n;
}

int main(void) {
    nanovec_math_init();

    float *vecs  = (float *)nanovec_aligned_alloc(64, (size_t)TEST_N * TEST_DIM * sizeof(float));
    float *recon = (float *)nanovec_aligned_alloc(64, (size_t)TEST_N * TEST_DIM * sizeof(float));
    float *query = (float *)nanovec_aligned_alloc(64, TEST_DIM * sizeof(float));
    CHECK(vecs && recon && query);

    generate_clustered_embeddings(vecs, TEST_N, TEST_DIM);
    for (uint32_t d = 0; d < TEST_DIM; ++d) {
        query[d] = vecs[d] + 0.02f * rng_normal();
    }
    normalize_vec(query, TEST_DIM);

    /* 1. SQ8 (per-dimension and global) */
    double sq8_pd_mse = 0.0, sq8_pd_cos = 0.0;
    double sq8_gl_mse = 0.0, sq8_gl_cos = 0.0;
    {
        uint8_t *codes = (uint8_t *)nanovec_aligned_alloc(64, (size_t)TEST_N * TEST_DIM);
        CHECK(codes != NULL);

        nanovec_sq8_t sq_pd;
        CHECK(nanovec_sq8_init(&sq_pd, TEST_DIM, true) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_train(&sq_pd, vecs, TEST_N) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_encode_batch(&sq_pd, vecs, codes, TEST_N) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_decode_batch(&sq_pd, codes, recon, TEST_N) == NANOVEC_SUCCESS);
        compute_distortion(vecs, recon, TEST_N, TEST_DIM, &sq8_pd_mse, &sq8_pd_cos);

        /* Verify SIMD asymmetric SQ8 dot accuracy and L2 parity */
        nanovec_sq8_query_t qctx;
        CHECK(nanovec_sq8_prepare_query(&sq_pd, query, &qctx) == NANOVEC_SUCCESS);
        double dot_mae = 0.0;
        for (size_t i = 0; i < TEST_N; ++i) {
            const uint8_t *c = codes + i * TEST_DIM;
            CHECK(nanovec_sq8_idot(c, qctx.wq, TEST_DIM) ==
                  nanovec_sq8_idot_scalar(c, qctx.wq, TEST_DIM));
            float approx_dot = nanovec_sq8_dist_dot_prepared(&qctx, c);
            float true_dot   = nanovec_dist_dot_fn(query, vecs + i * TEST_DIM, TEST_DIM);
            dot_mae += fabs((double)approx_dot - (double)true_dot);

            float sq8_l2 = nanovec_sq8_dist_l2(&sq_pd, query, c);
            float dec_l2 = nanovec_dist_l2_fn(query, recon + i * TEST_DIM, TEST_DIM);
            CHECK(fabsf(sq8_l2 - dec_l2) < 1e-5f);
        }
        dot_mae /= (double)TEST_N;
        CHECK(sq8_pd_mse < 1e-6);
        CHECK(sq8_pd_cos < 2e-4);
        CHECK(dot_mae < 5e-3);
        (void)printf(
            "  [SQ8-per_dim] ratio= 4.0x | MSE=%.3e | cos_distort=%.4e | dot_mae=%.3e\n",
            sq8_pd_mse, sq8_pd_cos, dot_mae
        );

        nanovec_sq8_query_destroy(&qctx);
        nanovec_sq8_destroy(&sq_pd);

        nanovec_sq8_t sq_gl;
        CHECK(nanovec_sq8_init(&sq_gl, TEST_DIM, false) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_train(&sq_gl, vecs, TEST_N) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_encode_batch(&sq_gl, vecs, codes, TEST_N) == NANOVEC_SUCCESS);
        CHECK(nanovec_sq8_decode_batch(&sq_gl, codes, recon, TEST_N) == NANOVEC_SUCCESS);
        compute_distortion(vecs, recon, TEST_N, TEST_DIM, &sq8_gl_mse, &sq8_gl_cos);
        CHECK(sq8_gl_mse < 5e-6);
        CHECK(sq8_gl_cos < 1e-3);
        (void)printf(
            "  [SQ8-global ] ratio= 4.0x | MSE=%.3e | cos_distort=%.4e\n",
            sq8_gl_mse, sq8_gl_cos
        );

        nanovec_sq8_destroy(&sq_gl);
        nanovec_aligned_free(codes);
    }

    /* 2. BQ (1-bit sign quantization) */
    double bq_mse = 0.0, bq_cos = 0.0;
    {
        uint32_t words = nanovec_bq_num_words(TEST_DIM);
        CHECK(nanovec_bq_byte_size(TEST_DIM) == 48U);
        uint64_t *bcodes = (uint64_t *)nanovec_aligned_alloc(64, (size_t)TEST_N * words * sizeof(uint64_t));
        uint64_t *bq     = (uint64_t *)nanovec_aligned_alloc(64, (size_t)words * sizeof(uint64_t));
        CHECK(bcodes && bq);

        CHECK(nanovec_bq_encode_batch(vecs, TEST_DIM, TEST_N, bcodes) == NANOVEC_SUCCESS);
        CHECK(nanovec_bq_encode(query, TEST_DIM, bq) == NANOVEC_SUCCESS);

        float inv_sqrt_d = 1.0f / sqrtf((float)TEST_DIM);
        for (size_t i = 0; i < TEST_N; ++i) {
            const uint64_t *bc = bcodes + i * words;
            CHECK(nanovec_bq_distance(bq, bc, words) == nanovec_bq_distance_scalar(bq, bc, words));
            CHECK(nanovec_bq_decode(bc, TEST_DIM, recon + i * TEST_DIM) == NANOVEC_SUCCESS);
            for (uint32_t d = 0; d < TEST_DIM; ++d) {
                recon[i * TEST_DIM + d] *= inv_sqrt_d;
            }
        }
        compute_distortion(vecs, recon, TEST_N, TEST_DIM, &bq_mse, &bq_cos);
        CHECK(bq_cos > 0.05 && bq_cos < 0.35);
        CHECK(bq_mse < 5e-3);
        (void)printf(
            "  [BQ-1bit    ] ratio=32.0x | MSE=%.3e | cos_distort=%.4e\n",
            bq_mse, bq_cos
        );

        nanovec_aligned_free(bcodes);
        nanovec_aligned_free(bq);
    }

    /* 3. PQ (M=48, sub_dim=8, 32x compression) + ADC */
    double pq_mse = 0.0, pq_cos = 0.0;
    {
        nanovec_pq_t pq;
        CHECK(nanovec_pq_init(&pq, TEST_DIM, TEST_PQ_M) == NANOVEC_SUCCESS);
        CHECK(nanovec_pq_code_size(&pq) == TEST_PQ_M);
        CHECK(nanovec_pq_train(&pq, vecs, TEST_N, 8, 33ULL) == NANOVEC_SUCCESS);

        uint8_t *pcodes = (uint8_t *)nanovec_aligned_alloc(64, (size_t)TEST_N * TEST_PQ_M);
        float   *adists = (float   *)nanovec_aligned_alloc(64, (size_t)TEST_N * sizeof(float));
        CHECK(pcodes && adists);

        CHECK(nanovec_pq_encode_batch(&pq, vecs, pcodes, TEST_N) == NANOVEC_SUCCESS);
        CHECK(nanovec_pq_decode_batch(&pq, pcodes, recon, TEST_N) == NANOVEC_SUCCESS);
        compute_distortion(vecs, recon, TEST_N, TEST_DIM, &pq_mse, &pq_cos);

        nanovec_adc_table_t tbl;
        CHECK(nanovec_adc_build(&pq, query, NANOVEC_METRIC_L2, &tbl) == NANOVEC_SUCCESS);
        nanovec_adc_distance_batch(&tbl, pcodes, TEST_N, adists);

        double adc_vs_fp32_mae = 0.0;
        for (size_t i = 0; i < TEST_N; ++i) {
            const uint8_t *code = pcodes + i * TEST_PQ_M;
            float s_dist = nanovec_adc_distance_scalar(&tbl, code);
            CHECK(fabsf(adists[i] - s_dist) < 1e-5f);

            float dec_l2  = nanovec_dist_l2_fn(query, recon + i * TEST_DIM, TEST_DIM);
            float true_l2 = nanovec_dist_l2_fn(query, vecs + i * TEST_DIM, TEST_DIM);
            CHECK(fabsf(adists[i] - dec_l2) < 1e-5f);
            adc_vs_fp32_mae += fabs((double)adists[i] - (double)true_l2);
        }
        adc_vs_fp32_mae /= (double)TEST_N;
        CHECK(pq_mse < 5e-4);
        CHECK(pq_cos < 0.15);
        CHECK(adc_vs_fp32_mae < 0.15);
        (void)printf(
            "  [PQ-M48+ADC ] ratio=32.0x | MSE=%.3e | cos_distort=%.4e | l2_mae=%.3e\n",
            pq_mse, pq_cos, adc_vs_fp32_mae
        );

        nanovec_adc_table_destroy(&tbl);
        nanovec_pq_destroy(&pq);
        nanovec_aligned_free(pcodes);
        nanovec_aligned_free(adists);
    }

    /* Statistical hierarchy: SQ8 < PQ < BQ distortion at 32x vs 4x compression */
    CHECK(sq8_pd_mse < pq_mse && pq_mse < bq_mse);
    CHECK(sq8_pd_cos < pq_cos && pq_cos < bq_cos);

    /* 4. Two-Stage Filtering Pipeline (Coarse BQ/PQ + Fine FP32 Re-ranking) */
    {
        const uint32_t TOP_K = 10U;
        uint64_t *vids = (uint64_t *)nanovec_aligned_alloc(64, (size_t)TEST_N * sizeof(uint64_t));
        uint64_t *bcodes = (uint64_t *)nanovec_aligned_alloc(
            64, (size_t)TEST_N * nanovec_bq_num_words(TEST_DIM) * sizeof(uint64_t)
        );
        uint8_t *pcodes = (uint8_t *)nanovec_aligned_alloc(64, (size_t)TEST_N * TEST_PQ_M);
        CHECK(vids && bcodes && pcodes);

        for (size_t i = 0; i < TEST_N; ++i) {
            vids[i] = 10000ULL + (uint64_t)i * 7ULL;
        }
        CHECK(nanovec_bq_encode_batch(vecs, TEST_DIM, TEST_N, bcodes) == NANOVEC_SUCCESS);

        nanovec_pq_t pq;
        CHECK(nanovec_pq_init(&pq, TEST_DIM, TEST_PQ_M) == NANOVEC_SUCCESS);
        CHECK(nanovec_pq_train(&pq, vecs, TEST_N, 8, 34ULL) == NANOVEC_SUCCESS);
        CHECK(nanovec_pq_encode_batch(&pq, vecs, pcodes, TEST_N) == NANOVEC_SUCCESS);

        /* Ground-truth exact top-K via full FP32 scan */
        nanovec_result_t gt[10];
        for (uint32_t i = 0; i < TOP_K; ++i) {
            gt[i].id = UINT64_MAX;
            gt[i].distance = 1e30f;
            gt[i]._pad = 0U;
        }
        for (size_t i = 0; i < TEST_N; ++i) {
            float d = nanovec_dist_l2_fn(query, vecs + i * TEST_DIM, TEST_DIM);
            if (d < gt[TOP_K - 1U].distance) {
                uint32_t pos = TOP_K - 1U;
                while (pos > 0U && d < gt[pos - 1U].distance) {
                    gt[pos] = gt[pos - 1U];
                    --pos;
                }
                gt[pos].id = vids[i];
                gt[pos].distance = d;
            }
        }

        nanovec_result_t bq_res[10], pq_res[10];
        size_t bq_cnt = 0, pq_cnt = 0;
        CHECK(nanovec_rerank_bq(
            query, TEST_DIM, bcodes, vecs, vids, TEST_N,
            NANOVEC_METRIC_L2, TOP_K, 10U, NULL, NULL, bq_res, &bq_cnt
        ) == NANOVEC_SUCCESS);
        CHECK(nanovec_rerank_pq(
            &pq, query, pcodes, vecs, vids, TEST_N,
            NANOVEC_METRIC_L2, TOP_K, 10U, NULL, NULL, pq_res, &pq_cnt
        ) == NANOVEC_SUCCESS);
        CHECK(bq_cnt == TOP_K && pq_cnt == TOP_K);

        /* Results sorted ascending and exact FP32 distances */
        for (uint32_t i = 1; i < TOP_K; ++i) {
            CHECK(bq_res[i].distance >= bq_res[i - 1U].distance);
            CHECK(pq_res[i].distance >= pq_res[i - 1U].distance);
        }

        uint32_t bq_hits = 0, pq_hits = 0;
        for (uint32_t i = 0; i < TOP_K; ++i) {
            for (uint32_t j = 0; j < TOP_K; ++j) {
                if (bq_res[i].id == gt[j].id) bq_hits++;
                if (pq_res[i].id == gt[j].id) pq_hits++;
            }
        }
        CHECK(bq_hits >= 8U);
        CHECK(pq_hits >= 9U);
        CHECK(pq_res[0].id == gt[0].id);

        (void)printf(
            "  [Two-Stage  ] Recall@10 (10x rerank): BQ+FP32=%.0f%% | PQ+FP32=%.0f%%\n",
            100.0 * (double)bq_hits / (double)TOP_K,
            100.0 * (double)pq_hits / (double)TOP_K
        );

        /* Edge cases: empty dataset, top_k=0, unsupported metric */
        size_t zcnt = 99;
        CHECK(nanovec_rerank_bq(
            query, TEST_DIM, bcodes, vecs, NULL, 0,
            NANOVEC_METRIC_L2, TOP_K, 10U, NULL, NULL, bq_res, &zcnt
        ) == NANOVEC_SUCCESS && zcnt == 0U);
        CHECK(nanovec_rerank_pq(
            &pq, query, pcodes, vecs, NULL, TEST_N,
            NANOVEC_METRIC_HAMMING, TOP_K, 10U, NULL, NULL, pq_res, &zcnt
        ) == NANOVEC_ERR_UNSUPPORTED);

        nanovec_pq_destroy(&pq);
        nanovec_aligned_free(vids);
        nanovec_aligned_free(bcodes);
        nanovec_aligned_free(pcodes);
    }

    nanovec_aligned_free(vecs);
    nanovec_aligned_free(recon);
    nanovec_aligned_free(query);

    (void)puts("PASS: tests/test_quant");
    return 0;
}
