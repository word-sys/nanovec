#include "math/distance.h"
#include "math/prefetch.h"

#include <immintrin.h>
#include <math.h>

float nanovec_dist_l2_avx512(const float *a, const float *b, uint32_t dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    uint32_t i = 0;
    uint32_t limit64 = dim & ~63U;
    for (; i < limit64; i += 64U) {
        if (i + 128U < dim) {
            nanovec_prefetch_l1(a + i + 128U);
            nanovec_prefetch_l1(b + i + 128U);
        }
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }

    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));

    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }

    float total = _mm512_reduce_add_ps(acc0);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float nanovec_dist_dot_avx512(const float *a, const float *b, uint32_t dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    uint32_t i = 0;
    uint32_t limit64 = dim & ~63U;
    for (; i < limit64; i += 64U) {
        if (i + 128U < dim) {
            nanovec_prefetch_l1(a + i + 128U);
            nanovec_prefetch_l1(b + i + 128U);
        }
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }

    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));

    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }

    float total = _mm512_reduce_add_ps(acc0);
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return 0.0f - total;
}

float nanovec_dist_cosine_avx512(const float *a, const float *b, uint32_t dim) {
    __m512 dot0 = _mm512_setzero_ps(), dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps(), dot3 = _mm512_setzero_ps();
    __m512 na0  = _mm512_setzero_ps(), na1  = _mm512_setzero_ps();
    __m512 na2  = _mm512_setzero_ps(), na3  = _mm512_setzero_ps();
    __m512 nb0  = _mm512_setzero_ps(), nb1  = _mm512_setzero_ps();
    __m512 nb2  = _mm512_setzero_ps(), nb3  = _mm512_setzero_ps();

    uint32_t i = 0;
    uint32_t limit64 = dim & ~63U;
    for (; i < limit64; i += 64U) {
        if (i + 128U < dim) {
            nanovec_prefetch_l1(a + i + 128U);
            nanovec_prefetch_l1(b + i + 128U);
        }
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        dot0 = _mm512_fmadd_ps(va0, vb0, dot0);
        na0  = _mm512_fmadd_ps(va0, va0, na0);
        nb0  = _mm512_fmadd_ps(vb0, vb0, nb0);

        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        dot1 = _mm512_fmadd_ps(va1, vb1, dot1);
        na1  = _mm512_fmadd_ps(va1, va1, na1);
        nb1  = _mm512_fmadd_ps(vb1, vb1, nb1);

        __m512 va2 = _mm512_loadu_ps(a + i + 32);
        __m512 vb2 = _mm512_loadu_ps(b + i + 32);
        dot2 = _mm512_fmadd_ps(va2, vb2, dot2);
        na2  = _mm512_fmadd_ps(va2, va2, na2);
        nb2  = _mm512_fmadd_ps(vb2, vb2, nb2);

        __m512 va3 = _mm512_loadu_ps(a + i + 48);
        __m512 vb3 = _mm512_loadu_ps(b + i + 48);
        dot3 = _mm512_fmadd_ps(va3, vb3, dot3);
        na3  = _mm512_fmadd_ps(va3, va3, na3);
        nb3  = _mm512_fmadd_ps(vb3, vb3, nb3);
    }

    dot0 = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    na0  = _mm512_add_ps(_mm512_add_ps(na0,  na1),  _mm512_add_ps(na2,  na3));
    nb0  = _mm512_add_ps(_mm512_add_ps(nb0,  nb1),  _mm512_add_ps(nb2,  nb3));

    uint32_t limit16 = dim & ~15U;
    for (; i < limit16; i += 16U) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        dot0 = _mm512_fmadd_ps(va, vb, dot0);
        na0  = _mm512_fmadd_ps(va, va, na0);
        nb0  = _mm512_fmadd_ps(vb, vb, nb0);
    }

    float dot    = _mm512_reduce_add_ps(dot0);
    float norm_a = _mm512_reduce_add_ps(na0);
    float norm_b = _mm512_reduce_add_ps(nb0);

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
