#include "math/distance.h"
#include "math/prefetch.h"

#if defined(__ARM_NEON) || defined(__aarch64__)

#ifndef NANOVEC_NEON_TEST_SHIM
#include <arm_neon.h>
#endif
#include <math.h>

#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(NANOVEC_NEON_TEST_SHIM)
#define NANOVEC_VFMAQ_F32(acc, x, y) vfmaq_f32((acc), (x), (y))
#else
#define NANOVEC_VFMAQ_F32(acc, x, y) vmlaq_f32((acc), (x), (y))
#endif

static inline float nanovec_hsum128_neon(float32x4_t v) {
#if defined(__aarch64__) || defined(NANOVEC_NEON_TEST_SHIM)
    return vaddvq_f32(v);
#else
    float32x2_t p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(p, p), 0);
#endif
}

float nanovec_dist_l2_neon(const float *a, const float *b, uint32_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        if (i + 32U < dim) {
            nanovec_prefetch_l1(a + i + 32U);
            nanovec_prefetch_l1(b + i + 32U);
        }
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = NANOVEC_VFMAQ_F32(acc0, d0, d0);
        acc1 = NANOVEC_VFMAQ_F32(acc1, d1, d1);
        acc2 = NANOVEC_VFMAQ_F32(acc2, d2, d2);
        acc3 = NANOVEC_VFMAQ_F32(acc3, d3, d3);
    }

    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

    uint32_t limit4 = dim & ~3U;
    for (; i < limit4; i += 4U) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = NANOVEC_VFMAQ_F32(acc0, d, d);
    }

    float total = nanovec_hsum128_neon(acc0);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float nanovec_dist_dot_neon(const float *a, const float *b, uint32_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        if (i + 32U < dim) {
            nanovec_prefetch_l1(a + i + 32U);
            nanovec_prefetch_l1(b + i + 32U);
        }
        acc0 = NANOVEC_VFMAQ_F32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = NANOVEC_VFMAQ_F32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = NANOVEC_VFMAQ_F32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = NANOVEC_VFMAQ_F32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

    uint32_t limit4 = dim & ~3U;
    for (; i < limit4; i += 4U) {
        acc0 = NANOVEC_VFMAQ_F32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    float total = nanovec_hsum128_neon(acc0);
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return 0.0f - total;
}

float nanovec_dist_cosine_neon(const float *a, const float *b, uint32_t dim) {
    float32x4_t dot0 = vdupq_n_f32(0.0f), dot1 = vdupq_n_f32(0.0f);
    float32x4_t dot2 = vdupq_n_f32(0.0f), dot3 = vdupq_n_f32(0.0f);
    float32x4_t na0  = vdupq_n_f32(0.0f), na1  = vdupq_n_f32(0.0f);
    float32x4_t na2  = vdupq_n_f32(0.0f), na3  = vdupq_n_f32(0.0f);
    float32x4_t nb0  = vdupq_n_f32(0.0f), nb1  = vdupq_n_f32(0.0f);
    float32x4_t nb2  = vdupq_n_f32(0.0f), nb3  = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        if (i + 32U < dim) {
            nanovec_prefetch_l1(a + i + 32U);
            nanovec_prefetch_l1(b + i + 32U);
        }
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        dot0 = NANOVEC_VFMAQ_F32(dot0, va0, vb0);
        na0  = NANOVEC_VFMAQ_F32(na0,  va0, va0);
        nb0  = NANOVEC_VFMAQ_F32(nb0,  vb0, vb0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        dot1 = NANOVEC_VFMAQ_F32(dot1, va1, vb1);
        na1  = NANOVEC_VFMAQ_F32(na1,  va1, va1);
        nb1  = NANOVEC_VFMAQ_F32(nb1,  vb1, vb1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        dot2 = NANOVEC_VFMAQ_F32(dot2, va2, vb2);
        na2  = NANOVEC_VFMAQ_F32(na2,  va2, va2);
        nb2  = NANOVEC_VFMAQ_F32(nb2,  vb2, vb2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        dot3 = NANOVEC_VFMAQ_F32(dot3, va3, vb3);
        na3  = NANOVEC_VFMAQ_F32(na3,  va3, va3);
        nb3  = NANOVEC_VFMAQ_F32(nb3,  vb3, vb3);
    }

    dot0 = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    na0  = vaddq_f32(vaddq_f32(na0,  na1),  vaddq_f32(na2,  na3));
    nb0  = vaddq_f32(vaddq_f32(nb0,  nb1),  vaddq_f32(nb2,  nb3));

    uint32_t limit4 = dim & ~3U;
    for (; i < limit4; i += 4U) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot0 = NANOVEC_VFMAQ_F32(dot0, va, vb);
        na0  = NANOVEC_VFMAQ_F32(na0,  va, va);
        nb0  = NANOVEC_VFMAQ_F32(nb0,  vb, vb);
    }

    float dot    = nanovec_hsum128_neon(dot0);
    float norm_a = nanovec_hsum128_neon(na0);
    float norm_b = nanovec_hsum128_neon(nb0);

    for (; i < dim; ++i) {
        float av = a[i];
        float bv = b[i];
        dot    += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

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

#else
typedef int nanovec_simd_neon_unused_tu_t;
#endif
