#include "quant/sq8.h"
#include "core/alloc.h"
#include "core/error.h"

#include <float.h>
#include <math.h>
#include <string.h>

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
