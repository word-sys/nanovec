#include "quant/bq.h"
#include "core/bitops.h"
#include "core/cpu.h"
#include "core/error.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void nanovec_bq_encode_scalar(const float *in, uint32_t dim, uint64_t *out) {
    uint32_t full_words = dim >> 6;
    for (uint32_t w = 0; w < full_words; ++w) {
        const float *p = in + (size_t)w * 64U;
        uint64_t word = 0ULL;
        for (uint32_t b = 0; b < 64U; b += 4U) {
            word |= ((uint64_t)(p[b]     >= 0.0f) << b)
                  | ((uint64_t)(p[b + 1] >= 0.0f) << (b + 1U))
                  | ((uint64_t)(p[b + 2] >= 0.0f) << (b + 2U))
                  | ((uint64_t)(p[b + 3] >= 0.0f) << (b + 3U));
        }
        out[w] = word;
    }

    uint32_t rem = dim & 63U;
    if (rem > 0U) {
        const float *p = in + (size_t)full_words * 64U;
        uint64_t word = 0ULL;
        for (uint32_t b = 0; b < rem; ++b) {
            word |= (uint64_t)(p[b] >= 0.0f) << b;
        }
        out[full_words] = word;
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static void nanovec_bq_encode_avx2(const float *in, uint32_t dim, uint64_t *out) {
    const __m256 zero = _mm256_setzero_ps();
    uint32_t full_words = dim >> 6;

    for (uint32_t w = 0; w < full_words; ++w) {
        const float *p = in + (size_t)w * 64U;
        uint64_t word = 0ULL;
        for (uint32_t chunk = 0; chunk < 8U; ++chunk) {
            __m256 v   = _mm256_loadu_ps(p + chunk * 8U);
            __m256 cmp = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
            uint64_t m = (uint64_t)(uint32_t)_mm256_movemask_ps(cmp);
            word |= m << (chunk * 8U);
        }
        out[w] = word;
    }

    uint32_t rem = dim & 63U;
    if (rem > 0U) {
        const float *p = in + (size_t)full_words * 64U;
        uint64_t word = 0ULL;
        uint32_t b = 0;
        for (; b + 8U <= rem; b += 8U) {
            __m256 v   = _mm256_loadu_ps(p + b);
            __m256 cmp = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
            uint64_t m = (uint64_t)(uint32_t)_mm256_movemask_ps(cmp);
            word |= m << b;
        }
        for (; b < rem; ++b) {
            word |= (uint64_t)(p[b] >= 0.0f) << b;
        }
        out[full_words] = word;
    }
}
#endif

nanovec_status_t nanovec_bq_encode(const float *in, uint32_t dim, uint64_t *out) {
    if (NANOVEC_UNLIKELY(in == NULL || out == NULL || dim == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "bq_encode requires non-null in/out and dim > 0"
        );
    }

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (nanovec_cpu_get_caps()->avx2) {
        nanovec_bq_encode_avx2(in, dim, out);
        return NANOVEC_SUCCESS;
    }
#endif
    nanovec_bq_encode_scalar(in, dim, out);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_bq_encode_batch(
    const float *in,
    uint32_t     dim,
    size_t       count,
    uint64_t    *out
) {
    if (NANOVEC_UNLIKELY(dim == 0 || (count > 0 && (in == NULL || out == NULL)))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "bq_encode_batch requires dim > 0 and valid buffers"
        );
    }

    uint32_t words = nanovec_bq_num_words(dim);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    bool use_avx2 = nanovec_cpu_get_caps()->avx2 != 0;
#endif
    for (size_t i = 0; i < count; ++i) {
        const float *src = in + i * dim;
        uint64_t    *dst = out + i * words;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if (use_avx2) {
            nanovec_bq_encode_avx2(src, dim, dst);
            continue;
        }
#endif
        nanovec_bq_encode_scalar(src, dim, dst);
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_bq_decode(const uint64_t *in, uint32_t dim, float *out) {
    if (NANOVEC_UNLIKELY(in == NULL || out == NULL || dim == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "bq_decode requires non-null in/out and dim > 0"
        );
    }

    for (uint32_t d = 0; d < dim; ++d) {
        uint64_t bit = (in[d >> 6] >> (d & 63U)) & 1ULL;
        out[d] = bit ? 1.0f : -1.0f;
    }
    return NANOVEC_SUCCESS;
}

uint32_t nanovec_bq_distance_scalar(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
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

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,popcnt")))
static uint32_t nanovec_bq_distance_avx2(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero     = _mm256_setzero_si256();
    __m256i total_v = _mm256_setzero_si256();

    uint32_t i = 0;
    uint32_t limit8 = num_words & ~7U;

    /* 64-byte chunks (8 uint64_t words): max 16 per byte lane, flush every 15 steps (<= 240). */
    while (i < limit8) {
        __m256i local = _mm256_setzero_si256();
        uint32_t batch_end = i + 120U;
        if (batch_end > limit8) {
            batch_end = limit8;
        }
        for (; i < batch_end; i += 8U) {
            __m256i x0 = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(const void *)(a + i)),
                _mm256_loadu_si256((const __m256i *)(const void *)(b + i))
            );
            __m256i x1 = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(const void *)(a + i + 4)),
                _mm256_loadu_si256((const __m256i *)(const void *)(b + i + 4))
            );

            __m256i p0 = _mm256_add_epi8(
                _mm256_shuffle_epi8(lut, _mm256_and_si256(x0, low_mask)),
                _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x0, 4), low_mask))
            );
            __m256i p1 = _mm256_add_epi8(
                _mm256_shuffle_epi8(lut, _mm256_and_si256(x1, low_mask)),
                _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x1, 4), low_mask))
            );
            local = _mm256_add_epi8(local, _mm256_add_epi8(p0, p1));
        }
        total_v = _mm256_add_epi64(total_v, _mm256_sad_epu8(local, zero));
    }

    if (i + 4U <= num_words) {
        __m256i x = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i *)(const void *)(a + i)),
            _mm256_loadu_si256((const __m256i *)(const void *)(b + i))
        );
        __m256i p = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(x, low_mask)),
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask))
        );
        total_v = _mm256_add_epi64(total_v, _mm256_sad_epu8(p, zero));
        i += 4U;
    }

    __m128i lo = _mm256_castsi256_si128(total_v);
    __m128i hi = _mm256_extracti128_si256(total_v, 1);
    __m128i s  = _mm_add_epi64(lo, hi);
    uint64_t total = (uint64_t)_mm_cvtsi128_si64(s)
                   + (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s, s));

    for (; i < num_words; ++i) {
        total += (uint64_t)nanovec_popcount64(a[i] ^ b[i]);
    }
    return (uint32_t)total;
}

__attribute__((target("avx512f,avx512bw,avx512dq,popcnt")))
static uint32_t nanovec_bq_distance_avx512(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    ));
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i zero     = _mm512_setzero_si512();
    __m512i total_v = _mm512_setzero_si512();

    uint32_t i = 0;
    uint32_t limit8 = num_words & ~7U;

    while (i < limit8) {
        __m512i local = _mm512_setzero_si512();
        uint32_t batch_end = i + 240U;
        if (batch_end > limit8) {
            batch_end = limit8;
        }
        for (; i < batch_end; i += 8U) {
            __m512i x = _mm512_xor_si512(
                _mm512_loadu_si512((const void *)(a + i)),
                _mm512_loadu_si512((const void *)(b + i))
            );
            __m512i p = _mm512_add_epi8(
                _mm512_shuffle_epi8(lut, _mm512_and_si512(x, low_mask)),
                _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(x, 4), low_mask))
            );
            local = _mm512_add_epi8(local, p);
        }
        total_v = _mm512_add_epi64(total_v, _mm512_sad_epu8(local, zero));
    }

    uint64_t total = (uint64_t)_mm512_reduce_add_epi64(total_v);
    for (; i < num_words; ++i) {
        total += (uint64_t)nanovec_popcount64(a[i] ^ b[i]);
    }
    return (uint32_t)total;
}
#elif defined(__aarch64__) && defined(__ARM_NEON)
static uint32_t nanovec_bq_distance_neon(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
    uint32x4_t total_v = vdupq_n_u32(0);
    uint32_t i = 0;
    uint32_t limit2 = num_words & ~1U;

    for (; i < limit2; i += 2U) {
        uint8x16_t va = vld1q_u8((const uint8_t *)(const void *)(a + i));
        uint8x16_t vb = vld1q_u8((const uint8_t *)(const void *)(b + i));
        uint8x16_t pc = vcntq_u8(veorq_u8(va, vb));
        total_v = vpadalq_u16(total_v, vpaddlq_u8(pc));
    }

    uint32_t total = vaddvq_u32(total_v);
    for (; i < num_words; ++i) {
        total += nanovec_popcount64(a[i] ^ b[i]);
    }
    return total;
}
#endif

uint32_t nanovec_bq_distance(const uint64_t *a, const uint64_t *b, uint32_t num_words) {
    if (NANOVEC_UNLIKELY(a == NULL || b == NULL || num_words == 0U)) {
        return 0U;
    }

    const nanovec_cpu_caps_t *caps = nanovec_cpu_get_caps();
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (caps->avx512f && caps->avx512bw && caps->avx512dq && caps->popcnt) {
        return nanovec_bq_distance_avx512(a, b, num_words);
    }
    if (caps->avx2 && caps->popcnt) {
        return nanovec_bq_distance_avx2(a, b, num_words);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    if (caps->neon || caps->asimd) {
        return nanovec_bq_distance_neon(a, b, num_words);
    }
#else
    (void)caps;
#endif
    return nanovec_bq_distance_scalar(a, b, num_words);
}
