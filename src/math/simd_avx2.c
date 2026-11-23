#include "math/distance.h"

#include <immintrin.h>
#include <math.h>

static inline float nanovec_hsum256_ps(__m256 v) {
    __m256 t = _mm256_hadd_ps(v, v);
    t = _mm256_hadd_ps(t, t);
    __m128 lo = _mm256_castps256_ps128(t);
    __m128 hi = _mm256_extractf128_ps(t, 1);
    return _mm_cvtss_f32(_mm_add_ss(lo, hi));
}

float nanovec_dist_l2_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    uint32_t i = 0;
    uint32_t limit32 = dim & ~31U;
    for (; i < limit32; i += 32U) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    uint32_t limit8 = dim & ~7U;
    for (; i < limit8; i += 8U) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }

    float total = nanovec_hsum256_ps(acc0);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float nanovec_dist_dot_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    uint32_t i = 0;
    uint32_t limit32 = dim & ~31U;
    for (; i < limit32; i += 32U) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    uint32_t limit8 = dim & ~7U;
    for (; i < limit8; i += 8U) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }

    float total = nanovec_hsum256_ps(acc0);
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return 0.0f - total;
}

float nanovec_dist_cosine_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 dot0 = _mm256_setzero_ps(), dot1 = _mm256_setzero_ps();
    __m256 dot2 = _mm256_setzero_ps(), dot3 = _mm256_setzero_ps();
    __m256 na0  = _mm256_setzero_ps(), na1  = _mm256_setzero_ps();
    __m256 na2  = _mm256_setzero_ps(), na3  = _mm256_setzero_ps();
    __m256 nb0  = _mm256_setzero_ps(), nb1  = _mm256_setzero_ps();
    __m256 nb2  = _mm256_setzero_ps(), nb3  = _mm256_setzero_ps();

    uint32_t i = 0;
    uint32_t limit32 = dim & ~31U;
    for (; i < limit32; i += 32U) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
        na0  = _mm256_fmadd_ps(va0, va0, na0);
        nb0  = _mm256_fmadd_ps(vb0, vb0, nb0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
        na1  = _mm256_fmadd_ps(va1, va1, na1);
        nb1  = _mm256_fmadd_ps(vb1, vb1, nb1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        dot2 = _mm256_fmadd_ps(va2, vb2, dot2);
        na2  = _mm256_fmadd_ps(va2, va2, na2);
        nb2  = _mm256_fmadd_ps(vb2, vb2, nb2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        dot3 = _mm256_fmadd_ps(va3, vb3, dot3);
        na3  = _mm256_fmadd_ps(va3, va3, na3);
        nb3  = _mm256_fmadd_ps(vb3, vb3, nb3);
    }

    dot0 = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    na0  = _mm256_add_ps(_mm256_add_ps(na0,  na1),  _mm256_add_ps(na2,  na3));
    nb0  = _mm256_add_ps(_mm256_add_ps(nb0,  nb1),  _mm256_add_ps(nb2,  nb3));

    uint32_t limit8 = dim & ~7U;
    for (; i < limit8; i += 8U) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va, vb, dot0);
        na0  = _mm256_fmadd_ps(va, va, na0);
        nb0  = _mm256_fmadd_ps(vb, vb, nb0);
    }

    float dot    = nanovec_hsum256_ps(dot0);
    float norm_a = nanovec_hsum256_ps(na0);
    float norm_b = nanovec_hsum256_ps(nb0);

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
