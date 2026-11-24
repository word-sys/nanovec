/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "math/distance.h"
#include "core/bitops.h"

#include <math.h>

float nanovec_dist_l2_scalar(const float *a, const float *b, uint32_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    uint32_t i = 0;
    uint32_t limit = dim & ~3U;
    for (; i < limit; i += 4U) {
        float d0 = a[i]     - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    float tail = 0.0f;
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        tail += d * d;
    }

    return ((sum0 + sum1) + (sum2 + sum3)) + tail;
}

float nanovec_dist_dot_scalar(const float *a, const float *b, uint32_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    uint32_t i = 0;
    uint32_t limit = dim & ~3U;
    for (; i < limit; i += 4U) {
        sum0 += a[i]     * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
    }

    float tail = 0.0f;
    for (; i < dim; ++i) {
        tail += a[i] * b[i];
    }

    float dot = ((sum0 + sum1) + (sum2 + sum3)) + tail;
    return 0.0f - dot;
}

float nanovec_dist_cosine_scalar(const float *a, const float *b, uint32_t dim) {
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float na0  = 0.0f, na1  = 0.0f, na2  = 0.0f, na3  = 0.0f;
    float nb0  = 0.0f, nb1  = 0.0f, nb2  = 0.0f, nb3  = 0.0f;

    uint32_t i = 0;
    uint32_t limit = dim & ~3U;
    for (; i < limit; i += 4U) {
        float a0 = a[i],     b0 = b[i];
        float a1 = a[i + 1], b1 = b[i + 1];
        float a2 = a[i + 2], b2 = b[i + 2];
        float a3 = a[i + 3], b3 = b[i + 3];

        dot0 += a0 * b0; na0 += a0 * a0; nb0 += b0 * b0;
        dot1 += a1 * b1; na1 += a1 * a1; nb1 += b1 * b1;
        dot2 += a2 * b2; na2 += a2 * a2; nb2 += b2 * b2;
        dot3 += a3 * b3; na3 += a3 * a3; nb3 += b3 * b3;
    }

    float dot_t = 0.0f, na_t = 0.0f, nb_t = 0.0f;
    for (; i < dim; ++i) {
        float av = a[i];
        float bv = b[i];
        dot_t += av * bv;
        na_t  += av * av;
        nb_t  += bv * bv;
    }

    float dot    = ((dot0 + dot1) + (dot2 + dot3)) + dot_t;
    float norm_a = ((na0 + na1) + (na2 + na3)) + na_t;
    float norm_b = ((nb0 + nb1) + (nb2 + nb3)) + nb_t;

    if (NANOVEC_UNLIKELY(norm_a <= 0.0f || norm_b <= 0.0f)) {
        return 1.0f;
    }

    double denom = sqrt((double)norm_a * (double)norm_b);
    if (NANOVEC_UNLIKELY(denom <= 0.0)) {
        return 1.0f;
    }

    double sim = (double)dot / denom;
    if (sim > 1.0) {
        sim = 1.0;
    } else if (sim < -1.0) {
        sim = -1.0;
    }
    return (float)(1.0 - sim);
}

uint32_t nanovec_dist_hamming_scalar(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    uint32_t i = 0;
    uint32_t limit = num_words & ~3U;
    for (; i < limit; i += 4U) {
        c0 += nanovec_popcount64(a[i]     ^ b[i]);
        c1 += nanovec_popcount64(a[i + 1] ^ b[i + 1]);
        c2 += nanovec_popcount64(a[i + 2] ^ b[i + 2]);
        c3 += nanovec_popcount64(a[i + 3] ^ b[i + 3]);
    }

    uint32_t tail = 0;
    for (; i < num_words; ++i) {
        tail += nanovec_popcount64(a[i] ^ b[i]);
    }

    return (c0 + c1) + (c2 + c3) + tail;
}
