/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_MATH_DISTANCE_H
#define NANOVEC_MATH_DISTANCE_H

#include "core/types.h"

/* All nanovec_dist_* kernels follow smaller-is-closer ordering. */
float    nanovec_dist_l2_scalar(const float *a, const float *b, uint32_t dim);
float    nanovec_dist_dot_scalar(const float *a, const float *b, uint32_t dim);
float    nanovec_dist_cosine_scalar(const float *a, const float *b, uint32_t dim);
uint32_t nanovec_dist_hamming_scalar(const uint64_t *a, const uint64_t *b, uint32_t num_words);

#if defined(__x86_64__) || defined(__i386__)
float nanovec_dist_l2_avx2(const float *a, const float *b, uint32_t dim);
float nanovec_dist_dot_avx2(const float *a, const float *b, uint32_t dim);
float nanovec_dist_cosine_avx2(const float *a, const float *b, uint32_t dim);

float nanovec_dist_l2_avx512(const float *a, const float *b, uint32_t dim);
float nanovec_dist_dot_avx512(const float *a, const float *b, uint32_t dim);
float nanovec_dist_cosine_avx512(const float *a, const float *b, uint32_t dim);
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
float nanovec_dist_l2_neon(const float *a, const float *b, uint32_t dim);
float nanovec_dist_dot_neon(const float *a, const float *b, uint32_t dim);
float nanovec_dist_cosine_neon(const float *a, const float *b, uint32_t dim);
#endif

typedef float    (*nanovec_dist_f32_fn_t)(const float *a, const float *b, uint32_t dim);
typedef uint32_t (*nanovec_dist_hamming_fn_t)(const uint64_t *a, const uint64_t *b, uint32_t num_words);

extern nanovec_dist_f32_fn_t     nanovec_dist_l2_fn;
extern nanovec_dist_f32_fn_t     nanovec_dist_dot_fn;
extern nanovec_dist_f32_fn_t     nanovec_dist_cosine_fn;
extern nanovec_dist_hamming_fn_t nanovec_dist_hamming_fn;

void                  nanovec_math_init(void);
nanovec_dist_f32_fn_t nanovec_get_dist_f32_fn(nanovec_metric_t metric);
float                 nanovec_compute_distance(
    nanovec_metric_t metric,
    const void      *a,
    const void      *b,
    uint32_t         dim
);

#endif /* NANOVEC_MATH_DISTANCE_H */
