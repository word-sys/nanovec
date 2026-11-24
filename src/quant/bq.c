#include "quant/bq.h"
#include "core/cpu.h"
#include "core/error.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
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
