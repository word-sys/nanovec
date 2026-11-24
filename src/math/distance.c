/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "math/distance.h"
#include "core/cpu.h"

#include <pthread.h>

nanovec_dist_f32_fn_t     nanovec_dist_l2_fn      = nanovec_dist_l2_scalar;
nanovec_dist_f32_fn_t     nanovec_dist_dot_fn     = nanovec_dist_dot_scalar;
nanovec_dist_f32_fn_t     nanovec_dist_cosine_fn  = nanovec_dist_cosine_scalar;
nanovec_dist_hamming_fn_t nanovec_dist_hamming_fn = nanovec_dist_hamming_scalar;

static pthread_once_t g_math_once = PTHREAD_ONCE_INIT;

static void nanovec_math_init_once(void) {
    const nanovec_cpu_caps_t *caps = nanovec_cpu_get_caps();

    nanovec_dist_l2_fn      = nanovec_dist_l2_scalar;
    nanovec_dist_dot_fn     = nanovec_dist_dot_scalar;
    nanovec_dist_cosine_fn  = nanovec_dist_cosine_scalar;
    nanovec_dist_hamming_fn = nanovec_dist_hamming_scalar;

#if defined(__x86_64__) || defined(__i386__)
    if (caps->avx512f && caps->avx512dq && caps->avx512bw) {
        nanovec_dist_l2_fn     = nanovec_dist_l2_avx512;
        nanovec_dist_dot_fn    = nanovec_dist_dot_avx512;
        nanovec_dist_cosine_fn = nanovec_dist_cosine_avx512;
    } else if (caps->avx2 && caps->fma) {
        nanovec_dist_l2_fn     = nanovec_dist_l2_avx2;
        nanovec_dist_dot_fn    = nanovec_dist_dot_avx2;
        nanovec_dist_cosine_fn = nanovec_dist_cosine_avx2;
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    if (caps->neon || caps->asimd) {
        nanovec_dist_l2_fn     = nanovec_dist_l2_neon;
        nanovec_dist_dot_fn    = nanovec_dist_dot_neon;
        nanovec_dist_cosine_fn = nanovec_dist_cosine_neon;
    }
#else
    (void)caps;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
void nanovec_math_init(void) {
    (void)pthread_once(&g_math_once, nanovec_math_init_once);
}

nanovec_dist_f32_fn_t nanovec_get_dist_f32_fn(nanovec_metric_t metric) {
    switch (metric) {
        case NANOVEC_METRIC_L2:     return nanovec_dist_l2_fn;
        case NANOVEC_METRIC_COSINE: return nanovec_dist_cosine_fn;
        case NANOVEC_METRIC_DOT:    return nanovec_dist_dot_fn;
        case NANOVEC_METRIC_HAMMING:
        default:                    return NULL;
    }
}

float nanovec_compute_distance(
    nanovec_metric_t metric,
    const void      *a,
    const void      *b,
    uint32_t         dim
) {
    switch (metric) {
        case NANOVEC_METRIC_L2:
            return nanovec_dist_l2_fn((const float *)a, (const float *)b, dim);
        case NANOVEC_METRIC_COSINE:
            return nanovec_dist_cosine_fn((const float *)a, (const float *)b, dim);
        case NANOVEC_METRIC_DOT:
            return nanovec_dist_dot_fn((const float *)a, (const float *)b, dim);
        case NANOVEC_METRIC_HAMMING: {
            uint32_t num_words = (dim + 63U) >> 6;
            return (float)nanovec_dist_hamming_fn(
                (const uint64_t *)a,
                (const uint64_t *)b,
                num_words
            );
        }
        default:
            return 0.0f;
    }
}
