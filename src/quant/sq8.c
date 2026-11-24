#include "quant/sq8.h"
#include "core/alloc.h"
#include "core/cpu.h"
#include "core/error.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

nanovec_status_t nanovec_sq8_init(nanovec_sq8_t *sq, uint16_t dim, bool per_dim) {
    if (NANOVEC_UNLIKELY(sq == NULL || dim == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_init requires non-null sq and dim > 0"
        );
    }

    memset(sq, 0, sizeof(*sq));

    sq->min_val   = (float *)nanovec_aligned_calloc(NANOVEC_CACHE_LINE, dim, sizeof(float));
    sq->max_val   = (float *)nanovec_aligned_calloc(NANOVEC_CACHE_LINE, dim, sizeof(float));
    sq->scale     = (float *)nanovec_aligned_calloc(NANOVEC_CACHE_LINE, dim, sizeof(float));
    sq->inv_scale = (float *)nanovec_aligned_calloc(NANOVEC_CACHE_LINE, dim, sizeof(float));

    if (NANOVEC_UNLIKELY(!sq->min_val || !sq->max_val || !sq->scale || !sq->inv_scale)) {
        nanovec_sq8_destroy(sq);
        return NANOVEC_ERR_OOM;
    }

    sq->dim     = dim;
    sq->per_dim = per_dim;
    sq->trained = false;
    return NANOVEC_SUCCESS;
}

void nanovec_sq8_destroy(nanovec_sq8_t *sq) {
    if (sq == NULL) {
        return;
    }
    nanovec_aligned_free(sq->min_val);
    nanovec_aligned_free(sq->max_val);
    nanovec_aligned_free(sq->scale);
    nanovec_aligned_free(sq->inv_scale);
    memset(sq, 0, sizeof(*sq));
}

static void nanovec_sq8_compute_scales(nanovec_sq8_t *sq) {
    for (uint32_t d = 0; d < sq->dim; ++d) {
        float range = sq->max_val[d] - sq->min_val[d];
        if (range > 0.0f && isfinite(range)) {
            sq->scale[d]     = range / 255.0f;
            sq->inv_scale[d] = 255.0f / range;
        } else {
            sq->scale[d]     = 0.0f;
            sq->inv_scale[d] = 0.0f;
        }
    }
}

nanovec_status_t nanovec_sq8_train(
    nanovec_sq8_t *sq,
    const float   *vectors,
    size_t         count
) {
    if (NANOVEC_UNLIKELY(sq == NULL || sq->dim == 0 || !sq->min_val || vectors == NULL || count == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_train requires initialized sq, non-null vectors, and count > 0"
        );
    }

    uint32_t dim = sq->dim;
    sq->trained  = false;

    if (sq->per_dim) {
        for (uint32_t d = 0; d < dim; ++d) {
            sq->min_val[d] = FLT_MAX;
            sq->max_val[d] = -FLT_MAX;
        }
        for (size_t i = 0; i < count; ++i) {
            const float *v = vectors + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                float x = v[d];
                if (NANOVEC_UNLIKELY(!isfinite(x))) {
                    return (nanovec_status_t)nanovec_set_error(
                        NANOVEC_ERR_INVALID_ARG,
                        "sq8_train encountered non-finite value at vec=%zu dim=%u",
                        i,
                        d
                    );
                }
                if (x < sq->min_val[d]) sq->min_val[d] = x;
                if (x > sq->max_val[d]) sq->max_val[d] = x;
            }
        }
    } else {
        float gmin = FLT_MAX;
        float gmax = -FLT_MAX;
        size_t total = count * dim;
        for (size_t i = 0; i < total; ++i) {
            float x = vectors[i];
            if (NANOVEC_UNLIKELY(!isfinite(x))) {
                return (nanovec_status_t)nanovec_set_error(
                    NANOVEC_ERR_INVALID_ARG,
                    "sq8_train encountered non-finite value at index=%zu",
                    i
                );
            }
            if (x < gmin) gmin = x;
            if (x > gmax) gmax = x;
        }
        for (uint32_t d = 0; d < dim; ++d) {
            sq->min_val[d] = gmin;
            sq->max_val[d] = gmax;
        }
    }

    nanovec_sq8_compute_scales(sq);
    sq->trained = true;
    return NANOVEC_SUCCESS;
}

static inline uint8_t nanovec_sq8_quantize_one(float x, float min_v, float inv_s) {
    if (NANOVEC_UNLIKELY(!isfinite(x) || inv_s == 0.0f)) {
        return 0U;
    }
    float q = floorf((x - min_v) * inv_s + 0.5f);
    if (q <= 0.0f) {
        return 0U;
    }
    if (q >= 255.0f) {
        return 255U;
    }
    return (uint8_t)q;
}

nanovec_status_t nanovec_sq8_encode(
    const nanovec_sq8_t *sq,
    const float         *in,
    uint8_t             *out
) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || in == NULL || out == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_encode requires trained sq and non-null in/out buffers"
        );
    }

    const float *min_v = sq->min_val;
    const float *inv_s = sq->inv_scale;
    uint32_t dim = sq->dim;

    for (uint32_t d = 0; d < dim; ++d) {
        out[d] = nanovec_sq8_quantize_one(in[d], min_v[d], inv_s[d]);
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_sq8_encode_batch(
    const nanovec_sq8_t *sq,
    const float         *in,
    uint8_t             *out,
    size_t               count
) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || (count > 0 && (in == NULL || out == NULL)))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_encode_batch requires trained sq and valid buffers"
        );
    }

    uint32_t dim = sq->dim;
    const float *min_v = sq->min_val;
    const float *inv_s = sq->inv_scale;

    for (size_t i = 0; i < count; ++i) {
        const float *src = in + i * dim;
        uint8_t     *dst = out + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            dst[d] = nanovec_sq8_quantize_one(src[d], min_v[d], inv_s[d]);
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_sq8_decode(
    const nanovec_sq8_t *sq,
    const uint8_t       *in,
    float               *out
) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || in == NULL || out == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_decode requires trained sq and non-null in/out buffers"
        );
    }

    const float *min_v = sq->min_val;
    const float *sc    = sq->scale;
    uint32_t dim = sq->dim;

    for (uint32_t d = 0; d < dim; ++d) {
        out[d] = min_v[d] + (float)in[d] * sc[d];
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_sq8_decode_batch(
    const nanovec_sq8_t *sq,
    const uint8_t       *in,
    float               *out,
    size_t               count
) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || (count > 0 && (in == NULL || out == NULL)))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_decode_batch requires trained sq and valid buffers"
        );
    }

    const float *min_v = sq->min_val;
    const float *sc    = sq->scale;
    uint32_t dim = sq->dim;

    for (size_t i = 0; i < count; ++i) {
        const uint8_t *src = in + i * dim;
        float         *dst = out + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            dst[d] = min_v[d] + (float)src[d] * sc[d];
        }
    }
    return NANOVEC_SUCCESS;
}

/* Query weights are quantized to [-64, +64] so u8*s8 pairwise sums (<= 32640) never saturate i16. */
#define NANOVEC_SQ8_QMAX 64.0f

int32_t nanovec_sq8_idot_scalar(const uint8_t *code, const int8_t *wq, uint32_t dim) {
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    uint32_t i = 0;
    uint32_t limit = dim & ~3U;

    for (; i < limit; i += 4U) {
        s0 += (int32_t)code[i]     * (int32_t)wq[i];
        s1 += (int32_t)code[i + 1] * (int32_t)wq[i + 1];
        s2 += (int32_t)code[i + 2] * (int32_t)wq[i + 2];
        s3 += (int32_t)code[i + 3] * (int32_t)wq[i + 3];
    }

    int32_t tail = 0;
    for (; i < dim; ++i) {
        tail += (int32_t)code[i] * (int32_t)wq[i];
    }
    return (s0 + s1) + (s2 + s3) + tail;
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static int32_t nanovec_sq8_idot_avx2(const uint8_t *code, const int8_t *wq, uint32_t dim) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);

    uint32_t i = 0;
    uint32_t limit64 = dim & ~63U;
    for (; i < limit64; i += 64U) {
        __m256i c0 = _mm256_loadu_si256((const __m256i *)(const void *)(code + i));
        __m256i w0 = _mm256_loadu_si256((const __m256i *)(const void *)(wq + i));
        __m256i p0 = _mm256_maddubs_epi16(c0, w0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));

        __m256i c1 = _mm256_loadu_si256((const __m256i *)(const void *)(code + i + 32));
        __m256i w1 = _mm256_loadu_si256((const __m256i *)(const void *)(wq + i + 32));
        __m256i p1 = _mm256_maddubs_epi16(c1, w1);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
    }

    acc0 = _mm256_add_epi32(acc0, acc1);

    uint32_t limit32 = dim & ~31U;
    for (; i < limit32; i += 32U) {
        __m256i c = _mm256_loadu_si256((const __m256i *)(const void *)(code + i));
        __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(wq + i));
        __m256i p = _mm256_maddubs_epi16(c, w);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p, ones));
    }

    __m128i lo = _mm256_castsi256_si128(acc0);
    __m128i hi = _mm256_extracti128_si256(acc0, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    int32_t total = _mm_cvtsi128_si32(s);

    for (; i < dim; ++i) {
        total += (int32_t)code[i] * (int32_t)wq[i];
    }
    return total;
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
static int32_t nanovec_sq8_idot_neon_dp(
    const uint8_t *code,
    const int8_t  *wq,
    uint32_t       dim,
    int32_t        wq_sum
) {
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t flip = vdupq_n_u8(0x80U);
    uint32_t i = 0;
    uint32_t limit16 = dim & ~15U;

    for (; i < limit16; i += 16U) {
        int8x16_t c_s8 = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(code + i), flip));
        int8x16_t w_s8 = vld1q_s8(wq + i);
        acc = vdotq_s32(acc, c_s8, w_s8);
    }

    int32_t total = vaddvq_s32(acc);
    int32_t tail_wsum = 0;
    for (; i < dim; ++i) {
        total += (int32_t)code[i] * (int32_t)wq[i];
        tail_wsum += (int32_t)wq[i];
    }
    return total + 128 * (wq_sum - tail_wsum);
}
#endif
static int32_t nanovec_sq8_idot_neon(const uint8_t *code, const int8_t *wq, uint32_t dim) {
    int32x4_t acc = vdupq_n_s32(0);
    uint32_t i = 0;
    uint32_t limit16 = dim & ~15U;

    for (; i < limit16; i += 16U) {
        uint8x16_t cu = vld1q_u8(code + i);
        int8x16_t  ws = vld1q_s8(wq + i);
        int16x8_t clo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cu)));
        int16x8_t chi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cu)));
        int16x8_t wlo = vmovl_s8(vget_low_s8(ws));
        int16x8_t whi = vmovl_s8(vget_high_s8(ws));
        acc = vmlal_s16(acc, vget_low_s16(clo),  vget_low_s16(wlo));
        acc = vmlal_s16(acc, vget_high_s16(clo), vget_high_s16(wlo));
        acc = vmlal_s16(acc, vget_low_s16(chi),  vget_low_s16(whi));
        acc = vmlal_s16(acc, vget_high_s16(chi), vget_high_s16(whi));
    }

    int32_t total = vaddvq_s32(acc);
    for (; i < dim; ++i) {
        total += (int32_t)code[i] * (int32_t)wq[i];
    }
    return total;
}
#endif

int32_t nanovec_sq8_idot(const uint8_t *code, const int8_t *wq, uint32_t dim) {
    const nanovec_cpu_caps_t *caps = nanovec_cpu_get_caps();
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (caps->avx2) {
        return nanovec_sq8_idot_avx2(code, wq, dim);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    if (caps->neon || caps->asimd) {
        return nanovec_sq8_idot_neon(code, wq, dim);
    }
#else
    (void)caps;
#endif
    return nanovec_sq8_idot_scalar(code, wq, dim);
}

static inline int8_t nanovec_sq8_quantize_qweight(float w, float inv_alpha) {
    float v = floorf(w * inv_alpha + 0.5f);
    if (v > NANOVEC_SQ8_QMAX)  return (int8_t)NANOVEC_SQ8_QMAX;
    if (v < -NANOVEC_SQ8_QMAX) return (int8_t)(-NANOVEC_SQ8_QMAX);
    return (int8_t)v;
}

static void nanovec_sq8_build_qweights(
    const nanovec_sq8_t *sq,
    const float         *query,
    int8_t              *wq,
    float               *out_base_dot,
    float               *out_alpha,
    int32_t             *out_wq_sum
) {
    uint32_t dim = sq->dim;
    const float *min_v = sq->min_val;
    const float *sc    = sq->scale;

    double base = 0.0;
    float  max_abs = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        base += (double)query[d] * (double)min_v[d];
        float a = fabsf(query[d] * sc[d]);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    *out_base_dot = (float)base;
    if (max_abs <= 0.0f || !isfinite(max_abs)) {
        memset(wq, 0, dim);
        *out_alpha  = 0.0f;
        *out_wq_sum = 0;
        return;
    }

    float alpha     = max_abs / NANOVEC_SQ8_QMAX;
    float inv_alpha = NANOVEC_SQ8_QMAX / max_abs;
    int32_t wsum    = 0;

    for (uint32_t d = 0; d < dim; ++d) {
        int8_t qv = nanovec_sq8_quantize_qweight(query[d] * sc[d], inv_alpha);
        wq[d] = qv;
        wsum += (int32_t)qv;
    }

    *out_alpha  = alpha;
    *out_wq_sum = wsum;
}

nanovec_status_t nanovec_sq8_prepare_query(
    const nanovec_sq8_t *sq,
    const float         *query,
    nanovec_sq8_query_t *out_q
) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || query == NULL || out_q == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "sq8_prepare_query requires trained sq, non-null query and out_q"
        );
    }

    memset(out_q, 0, sizeof(*out_q));
    out_q->wq = (int8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, sq->dim);
    if (NANOVEC_UNLIKELY(out_q->wq == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    out_q->dim = sq->dim;
    nanovec_sq8_build_qweights(
        sq,
        query,
        out_q->wq,
        &out_q->base_dot,
        &out_q->alpha,
        &out_q->wq_sum
    );
    return NANOVEC_SUCCESS;
}

void nanovec_sq8_query_destroy(nanovec_sq8_query_t *q) {
    if (q == NULL) {
        return;
    }
    nanovec_aligned_free(q->wq);
    memset(q, 0, sizeof(*q));
}

float nanovec_sq8_dot_prepared(const nanovec_sq8_query_t *q, const uint8_t *code) {
    if (NANOVEC_UNLIKELY(q == NULL || q->wq == NULL || code == NULL)) {
        return 0.0f;
    }
    if (NANOVEC_UNLIKELY(q->alpha == 0.0f)) {
        return q->base_dot;
    }
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32_t idot = nanovec_sq8_idot_neon_dp(code, q->wq, q->dim, q->wq_sum);
#else
    int32_t idot = nanovec_sq8_idot(code, q->wq, q->dim);
#endif
    return q->base_dot + q->alpha * (float)idot;
}

float nanovec_sq8_dist_dot_prepared(const nanovec_sq8_query_t *q, const uint8_t *code) {
    return 0.0f - nanovec_sq8_dot_prepared(q, code);
}

float nanovec_sq8_dist_dot(const nanovec_sq8_t *sq, const float *query, const uint8_t *code) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || query == NULL || code == NULL)) {
        return 0.0f;
    }

    int8_t stack_wq[2048];
    int8_t *wq = stack_wq;
    if (NANOVEC_UNLIKELY(sq->dim > sizeof(stack_wq))) {
        wq = (int8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, sq->dim);
        if (wq == NULL) {
            return 0.0f;
        }
    }

    float   base_dot = 0.0f;
    float   alpha    = 0.0f;
    int32_t wq_sum   = 0;
    nanovec_sq8_build_qweights(sq, query, wq, &base_dot, &alpha, &wq_sum);
    (void)wq_sum;

    float dot = base_dot;
    if (alpha != 0.0f) {
        dot += alpha * (float)nanovec_sq8_idot(code, wq, sq->dim);
    }

    if (wq != stack_wq) {
        nanovec_aligned_free(wq);
    }
    return 0.0f - dot;
}

float nanovec_sq8_dist_l2(const nanovec_sq8_t *sq, const float *query, const uint8_t *code) {
    if (NANOVEC_UNLIKELY(sq == NULL || !sq->trained || query == NULL || code == NULL)) {
        return 0.0f;
    }

    const float *min_v = sq->min_val;
    const float *sc    = sq->scale;
    uint32_t dim = sq->dim;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint32_t i = 0;
    uint32_t limit = dim & ~3U;

    for (; i < limit; i += 4U) {
        float d0 = query[i]     - (min_v[i]     + (float)code[i]     * sc[i]);
        float d1 = query[i + 1] - (min_v[i + 1] + (float)code[i + 1] * sc[i + 1]);
        float d2 = query[i + 2] - (min_v[i + 2] + (float)code[i + 2] * sc[i + 2]);
        float d3 = query[i + 3] - (min_v[i + 3] + (float)code[i + 3] * sc[i + 3]);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    float tail = 0.0f;
    for (; i < dim; ++i) {
        float d = query[i] - (min_v[i] + (float)code[i] * sc[i]);
        tail += d * d;
    }
    return ((s0 + s1) + (s2 + s3)) + tail;
}
