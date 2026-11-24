/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "core/alloc.h"
#include "core/cpu.h"
#include "math/distance.h"
#include "math/prefetch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON_ABS 1e-5f
#define NUM_VECTORS 100000U
#define BENCH_DIM   128U

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static uint64_t g_rng = 0x4E414E4F56454301ULL;

static uint64_t rng_next_u64(void) {
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_f32_signed(void) {
    uint32_t u = (uint32_t)(rng_next_u64() >> 40);
    return ((float)u / (float)(1U << 24)) * 2.0f - 1.0f;
}

static void fill_unit_vec(float *v, uint32_t dim) {
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        float x = rng_f32_signed();
        v[i] = x;
        sum_sq += (double)x * (double)x;
    }
    if (sum_sq > 0.0) {
        float inv = (float)(1.0 / sqrt(sum_sq));
        for (uint32_t i = 0; i < dim; ++i) {
            v[i] *= inv;
        }
    }
}

static void verify_backend_pair(
    const char            *name,
    nanovec_dist_f32_fn_t  l2_fn,
    nanovec_dist_f32_fn_t  dot_fn,
    nanovec_dist_f32_fn_t  cos_fn,
    const float           *vecs,
    uint32_t               count,
    uint32_t               dim
) {
    float max_l2  = 0.0f;
    float max_dot = 0.0f;
    float max_cos = 0.0f;

    const float *query = vecs;
    for (uint32_t i = 0; i < count; ++i) {
        const float *v = vecs + (size_t)i * dim;
        if (i + 1U < count) {
            nanovec_prefetch_l1(vecs + (size_t)(i + 1U) * dim);
        }

        float ref_l2  = nanovec_dist_l2_scalar(query, v, dim);
        float got_l2  = l2_fn(query, v, dim);
        float err_l2  = fabsf(got_l2 - ref_l2);
        if (err_l2 > max_l2) max_l2 = err_l2;
        CHECK(err_l2 < EPSILON_ABS);

        float ref_dot = nanovec_dist_dot_scalar(query, v, dim);
        float got_dot = dot_fn(query, v, dim);
        float err_dot = fabsf(got_dot - ref_dot);
        if (err_dot > max_dot) max_dot = err_dot;
        CHECK(err_dot < EPSILON_ABS);

        float ref_cos = nanovec_dist_cosine_scalar(query, v, dim);
        float got_cos = cos_fn(query, v, dim);
        float err_cos = fabsf(got_cos - ref_cos);
        if (err_cos > max_cos) max_cos = err_cos;
        CHECK(err_cos < EPSILON_ABS);
        CHECK(got_cos >= 0.0f && got_cos <= 2.0f);
    }

    (void)printf(
        "  [%s] %u vectors (dim=%u): max_abs_err l2=%.2e dot=%.2e cos=%.2e\n",
        name, count, dim, (double)max_l2, (double)max_dot, (double)max_cos
    );
}

static void verify_dims_and_alignment(
    nanovec_dist_f32_fn_t l2_fn,
    nanovec_dist_f32_fn_t dot_fn,
    nanovec_dist_f32_fn_t cos_fn
) {
    const uint32_t dims[] = {
        0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 384, 768, 1536
    };

    float *buf_a = (float *)nanovec_aligned_alloc(64, (1536 + 8) * sizeof(float));
    float *buf_b = (float *)nanovec_aligned_alloc(64, (1536 + 8) * sizeof(float));
    CHECK(buf_a != NULL && buf_b != NULL);

    for (size_t di = 0; di < NANOVEC_ARRAY_LEN(dims); ++di) {
        uint32_t d = dims[di];
        const size_t offsets[] = {0, 1, 3};
        for (size_t oi = 0; oi < NANOVEC_ARRAY_LEN(offsets); ++oi) {
            float *a = buf_a + offsets[oi];
            float *b = buf_b + offsets[oi];
            for (int t = 0; t < 32; ++t) {
                fill_unit_vec(a, d);
                fill_unit_vec(b, d);

                CHECK(l2_fn(a, a, d) == 0.0f);
                if (d > 0) {
                    CHECK(cos_fn(a, a, d) == 0.0f);
                } else {
                    CHECK(cos_fn(a, b, d) == 1.0f);
                    CHECK(dot_fn(a, b, d) == 0.0f);
                }

                CHECK(fabsf(l2_fn(a, b, d)  - nanovec_dist_l2_scalar(a, b, d))     < EPSILON_ABS);
                CHECK(fabsf(dot_fn(a, b, d) - nanovec_dist_dot_scalar(a, b, d))    < EPSILON_ABS);
                CHECK(fabsf(cos_fn(a, b, d) - nanovec_dist_cosine_scalar(a, b, d)) < EPSILON_ABS);
            }
        }
    }

    nanovec_aligned_free(buf_a);
    nanovec_aligned_free(buf_b);
}

static void verify_edge_identities(void) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float n[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float o2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float big[4] = {1e15f, -2e15f, 3e15f, -4e15f};

    CHECK(nanovec_compute_distance(NANOVEC_METRIC_L2, a, a, 4) == 0.0f);
    CHECK(fabsf(nanovec_compute_distance(NANOVEC_METRIC_L2, a, b, 4) - 30.0f) < EPSILON_ABS);
    CHECK(fabsf(nanovec_compute_distance(NANOVEC_METRIC_DOT, a, b, 4) - (-60.0f)) < EPSILON_ABS);
    CHECK(nanovec_compute_distance(NANOVEC_METRIC_DOT, o1, o2, 4) == 0.0f);
    CHECK(nanovec_compute_distance(NANOVEC_METRIC_COSINE, a, a, 4) == 0.0f);
    CHECK(fabsf(nanovec_compute_distance(NANOVEC_METRIC_COSINE, a, b, 4)) < EPSILON_ABS);
    CHECK(fabsf(nanovec_compute_distance(NANOVEC_METRIC_COSINE, a, n, 4) - 2.0f) < EPSILON_ABS);
    CHECK(fabsf(nanovec_compute_distance(NANOVEC_METRIC_COSINE, o1, o2, 4) - 1.0f) < EPSILON_ABS);
    CHECK(nanovec_compute_distance(NANOVEC_METRIC_COSINE, z, a, 4) == 1.0f);
    CHECK(nanovec_compute_distance(NANOVEC_METRIC_COSINE, a, z, 4) == 1.0f);
    CHECK(nanovec_compute_distance(NANOVEC_METRIC_COSINE, big, big, 4) == 0.0f);
}

static void verify_hamming(void) {
    uint64_t a[24], b[24], inv[24];
    for (uint32_t trial = 0; trial < 2000; ++trial) {
        uint32_t n = trial % 25U;
        uint32_t expected = 0;
        for (uint32_t w = 0; w < n; ++w) {
            a[w] = rng_next_u64();
            b[w] = rng_next_u64();
            inv[w] = ~a[w];
            uint64_t x = a[w] ^ b[w];
            for (int bit = 0; bit < 64; ++bit) {
                expected += (uint32_t)((x >> bit) & 1ULL);
            }
        }
        uint32_t got = nanovec_dist_hamming_fn(a, b, n);
        CHECK(got == expected);
        CHECK(nanovec_dist_hamming_fn(b, a, n) == expected);
        CHECK(nanovec_dist_hamming_fn(a, a, n) == 0U);
        CHECK(nanovec_dist_hamming_fn(a, inv, n) == n * 64U);
        CHECK(nanovec_compute_distance(NANOVEC_METRIC_HAMMING, a, b, n * 64U) == (float)expected);
    }
}

int main(void) {
    nanovec_math_init();
    const nanovec_cpu_caps_t *caps = nanovec_cpu_get_caps();

    (void)printf("test_math: host isa=%s\n", nanovec_cpu_best_isa_name());

    verify_edge_identities();
    verify_hamming();

    float *vecs = (float *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        (size_t)NUM_VECTORS * BENCH_DIM * sizeof(float)
    );
    CHECK(vecs != NULL);

    for (uint32_t i = 0; i < NUM_VECTORS; ++i) {
        fill_unit_vec(vecs + (size_t)i * BENCH_DIM, BENCH_DIM);
    }

    verify_backend_pair(
        "dispatched",
        nanovec_dist_l2_fn,
        nanovec_dist_dot_fn,
        nanovec_dist_cosine_fn,
        vecs,
        NUM_VECTORS,
        BENCH_DIM
    );
    verify_dims_and_alignment(nanovec_dist_l2_fn, nanovec_dist_dot_fn, nanovec_dist_cosine_fn);

#if defined(__x86_64__) || defined(__i386__)
    if (caps->avx2 && caps->fma) {
        verify_backend_pair(
            "avx2",
            nanovec_dist_l2_avx2,
            nanovec_dist_dot_avx2,
            nanovec_dist_cosine_avx2,
            vecs,
            NUM_VECTORS,
            BENCH_DIM
        );
        verify_dims_and_alignment(
            nanovec_dist_l2_avx2,
            nanovec_dist_dot_avx2,
            nanovec_dist_cosine_avx2
        );
    }
    if (caps->avx512f && caps->avx512dq && caps->avx512bw) {
        verify_backend_pair(
            "avx512",
            nanovec_dist_l2_avx512,
            nanovec_dist_dot_avx512,
            nanovec_dist_cosine_avx512,
            vecs,
            NUM_VECTORS,
            BENCH_DIM
        );
        verify_dims_and_alignment(
            nanovec_dist_l2_avx512,
            nanovec_dist_dot_avx512,
            nanovec_dist_cosine_avx512
        );
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    if (caps->neon || caps->asimd) {
        verify_backend_pair(
            "neon",
            nanovec_dist_l2_neon,
            nanovec_dist_dot_neon,
            nanovec_dist_cosine_neon,
            vecs,
            NUM_VECTORS,
            BENCH_DIM
        );
        verify_dims_and_alignment(
            nanovec_dist_l2_neon,
            nanovec_dist_dot_neon,
            nanovec_dist_cosine_neon
        );
    }
#else
    (void)caps;
#endif

    nanovec_aligned_free(vecs);
    (void)puts("PASS: tests/test_math");
    return 0;
}
