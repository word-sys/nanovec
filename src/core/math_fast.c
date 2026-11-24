/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "core/math_fast.h"

#include <math.h>
#include <string.h>

#define NANOVEC_INV_LN2   1.44269504088896341f
#define NANOVEC_LN2_HI    0.693359375f
#define NANOVEC_LN2_LO   (-2.12194440e-4f)
#define NANOVEC_SQRT2     1.41421356237309505f
#define NANOVEC_EXP_MIN  (-87.33654475f)
#define NANOVEC_EXP_MAX   88.0f
#define NANOVEC_FLT_MIN   1.17549435e-38f

static inline uint32_t nanovec_f32_to_u32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static inline float nanovec_u32_to_f32(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float nanovec_fast_rsqrt_impl(float x) {
    if (NANOVEC_UNLIKELY(isnan(x) || x < 0.0f)) {
        return NAN;
    }
    if (NANOVEC_UNLIKELY(x == 0.0f)) {
        return INFINITY;
    }
    if (NANOVEC_UNLIKELY(isinf(x))) {
        return 0.0f;
    }

    float scale = 1.0f;
    if (NANOVEC_UNLIKELY(x < NANOVEC_FLT_MIN)) {
        x *= 16777216.0f;
        scale = 4096.0f;
    }

    float    half_x = 0.5f * x;
    uint32_t i      = 0x5F375A86U - (nanovec_f32_to_u32(x) >> 1);
    float    y      = nanovec_u32_to_f32(i);

    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return y * scale;
}

static inline float nanovec_fast_exp_impl(float x) {
    if (NANOVEC_UNLIKELY(isnan(x))) {
        return NAN;
    }
    if (NANOVEC_UNLIKELY(x <= NANOVEC_EXP_MIN)) {
        return 0.0f;
    }
    if (NANOVEC_UNLIKELY(x >= NANOVEC_EXP_MAX)) {
        return x >= 88.72283905f ? INFINITY : expf(x);
    }

    float   kf = floorf(x * NANOVEC_INV_LN2 + 0.5f);
    int32_t k  = (int32_t)kf;
    float   r  = (x - kf * NANOVEC_LN2_HI) - kf * NANOVEC_LN2_LO;

    float p = 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 5.00000000e-1f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    float pow2 = nanovec_u32_to_f32((uint32_t)(k + 127) << 23);
    return p * pow2;
}

static inline float nanovec_fast_log_impl(float x) {
    if (NANOVEC_UNLIKELY(isnan(x) || x < 0.0f)) {
        return NAN;
    }
    if (NANOVEC_UNLIKELY(x == 0.0f)) {
        return -INFINITY;
    }
    if (NANOVEC_UNLIKELY(isinf(x))) {
        return INFINITY;
    }

    int32_t exp_adj = 0;
    if (NANOVEC_UNLIKELY(x < NANOVEC_FLT_MIN)) {
        x *= 16777216.0f;
        exp_adj = -24;
    }

    uint32_t bits = nanovec_f32_to_u32(x);
    int32_t  e    = (int32_t)((bits >> 23) & 0xFFU) - 127 + exp_adj;
    float    m    = nanovec_u32_to_f32((bits & 0x007FFFFFU) | 0x3F800000U);

    if (m > NANOVEC_SQRT2) {
        m *= 0.5f;
        e += 1;
    }

    float z  = (m - 1.0f) / (m + 1.0f);
    float z2 = z * z;
    float p  = 1.11111111e-1f;
    p = p * z2 + 1.42857143e-1f;
    p = p * z2 + 2.00000000e-1f;
    p = p * z2 + 3.33333333e-1f;
    p = p * z2 + 1.0f;

    float ef = (float)e;
    return (2.0f * z * p) + ef * NANOVEC_LN2_LO + ef * NANOVEC_LN2_HI;
}

float nanovec_fast_rsqrt(float x) { return nanovec_fast_rsqrt_impl(x); }
float nanovec_fast_exp(float x)   { return nanovec_fast_exp_impl(x); }
float nanovec_fast_log(float x)   { return nanovec_fast_log_impl(x); }

float nanovec_fast_tanh(float x) {
    if (NANOVEC_UNLIKELY(isnan(x))) {
        return NAN;
    }
    if (x >= 9.0f) {
        return 1.0f;
    }
    if (x <= -9.0f) {
        return -1.0f;
    }
    if (x >= 0.0f) {
        float e = nanovec_fast_exp_impl(-2.0f * x);
        return (1.0f - e) / (1.0f + e);
    }
    float e = nanovec_fast_exp_impl(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

float nanovec_fast_sigmoid(float x) {
    if (NANOVEC_UNLIKELY(isnan(x))) {
        return NAN;
    }
    if (x >= 0.0f) {
        return 1.0f / (1.0f + nanovec_fast_exp_impl(-x));
    }
    float e = nanovec_fast_exp_impl(x);
    return e / (1.0f + e);
}

void nanovec_fast_rsqrt_vec(const float *in, float *out, size_t n) {
    if (NANOVEC_UNLIKELY(in == NULL || out == NULL)) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = nanovec_fast_rsqrt_impl(in[i]);
    }
}

void nanovec_fast_exp_vec(const float *in, float *out, size_t n) {
    if (NANOVEC_UNLIKELY(in == NULL || out == NULL)) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = nanovec_fast_exp_impl(in[i]);
    }
}

void nanovec_fast_log_vec(const float *in, float *out, size_t n) {
    if (NANOVEC_UNLIKELY(in == NULL || out == NULL)) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = nanovec_fast_log_impl(in[i]);
    }
}
