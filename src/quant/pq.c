/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "quant/pq.h"
#include "core/alloc.h"
#include "core/error.h"
#include "quant/kmeans.h"

#include <math.h>
#include <string.h>

#define NANOVEC_PQ_DEFAULT_SEED   0x5051545241494E01ULL
#define NANOVEC_PQ_SERIAL_HDR     64U
#define NANOVEC_PQ_SERIAL_VERSION 1U
#define NANOVEC_PQ_FLAG_TRAINED   (1U << 0)

nanovec_status_t nanovec_pq_init(nanovec_pq_t *pq, uint16_t dim, uint16_t m) {
    if (NANOVEC_UNLIKELY(pq == NULL || dim == 0 || m == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_init requires non-null pq, dim > 0, and m > 0"
        );
    }
    if (NANOVEC_UNLIKELY((dim % m) != 0U)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_init requires dim (%u) divisible by m (%u)",
            (unsigned)dim,
            (unsigned)m
        );
    }

    memset(pq, 0, sizeof(*pq));

    size_t total_floats = (size_t)NANOVEC_PQ_K * (size_t)dim;
    float *centroids = (float *)nanovec_aligned_calloc(
        NANOVEC_CACHE_LINE,
        total_floats,
        sizeof(float)
    );
    if (NANOVEC_UNLIKELY(centroids == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    pq->dim            = dim;
    pq->m              = m;
    pq->sub_dim        = (uint16_t)(dim / m);
    pq->ksub           = (uint16_t)NANOVEC_PQ_K;
    pq->trained        = false;
    pq->owns_centroids = true;
    pq->centroids      = centroids;
    return NANOVEC_SUCCESS;
}

void nanovec_pq_destroy(nanovec_pq_t *pq) {
    if (pq == NULL) {
        return;
    }
    if (pq->owns_centroids) {
        nanovec_aligned_free(pq->centroids);
    }
    memset(pq, 0, sizeof(*pq));
}

nanovec_status_t nanovec_pq_train(
    nanovec_pq_t *pq,
    const float  *vectors,
    size_t        count,
    uint32_t      max_iters,
    uint64_t      seed
) {
    if (NANOVEC_UNLIKELY(
            pq == NULL || pq->centroids == NULL || pq->dim == 0 || pq->m == 0 ||
            vectors == NULL || count == 0
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_train requires initialized pq, non-null vectors, and count > 0"
        );
    }

    pq->trained = false;

    uint32_t dim     = pq->dim;
    uint32_t m       = pq->m;
    uint32_t sub_dim = pq->sub_dim;
    uint32_t ksub    = pq->ksub;

    if (NANOVEC_UNLIKELY(count > SIZE_MAX / ((size_t)sub_dim * sizeof(float)))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_OOM, "pq_train buffer overflow");
    }

    float *sub_buf = (float *)nanovec_aligned_alloc(
        NANOVEC_CACHE_LINE,
        count * (size_t)sub_dim * sizeof(float)
    );
    if (NANOVEC_UNLIKELY(sub_buf == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    uint64_t base_seed = seed != 0ULL ? seed : NANOVEC_PQ_DEFAULT_SEED;
    size_t sub_bytes   = (size_t)sub_dim * sizeof(float);

    for (uint32_t s = 0; s < m; ++s) {
        size_t col_off = (size_t)s * sub_dim;
        for (size_t i = 0; i < count; ++i) {
            memcpy(
                sub_buf + i * sub_dim,
                vectors + i * dim + col_off,
                sub_bytes
            );
        }

        nanovec_kmeans_params_t kp = {
            .k         = ksub,
            .max_iters = max_iters,
            .tol       = 1e-5f,
            .seed      = base_seed + (uint64_t)s * 0x9E3779B97F4A7C15ULL,
            .spherical = false
        };

        float *dst = pq->centroids + ((size_t)s * ksub * sub_dim);
        nanovec_status_t st = nanovec_kmeans_fit(sub_buf, count, sub_dim, &kp, dst, NULL, NULL);
        if (NANOVEC_UNLIKELY(st != NANOVEC_SUCCESS)) {
            nanovec_aligned_free(sub_buf);
            return st;
        }
    }

    nanovec_aligned_free(sub_buf);
    pq->trained = true;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_encode(
    const nanovec_pq_t *pq,
    const float        *in,
    uint8_t            *out_code
) {
    if (NANOVEC_UNLIKELY(pq == NULL || !pq->trained || in == NULL || out_code == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_encode requires trained pq and non-null in/out_code"
        );
    }

    uint32_t m       = pq->m;
    uint32_t sub_dim = pq->sub_dim;
    uint32_t ksub    = pq->ksub;

    for (uint32_t s = 0; s < m; ++s) {
        const float *sub_vec   = in + (size_t)s * sub_dim;
        const float *sub_cents = pq->centroids + ((size_t)s * ksub * sub_dim);
        uint32_t best = nanovec_kmeans_assign_one(sub_vec, sub_cents, ksub, sub_dim, NULL);
        out_code[s] = (uint8_t)best;
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_encode_batch(
    const nanovec_pq_t *pq,
    const float        *in,
    uint8_t            *out_codes,
    size_t              count
) {
    if (NANOVEC_UNLIKELY(
            pq == NULL || !pq->trained || (count > 0 && (in == NULL || out_codes == NULL))
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_encode_batch requires trained pq and valid buffers"
        );
    }

    uint32_t dim     = pq->dim;
    uint32_t m       = pq->m;
    uint32_t sub_dim = pq->sub_dim;
    uint32_t ksub    = pq->ksub;

    /* Subspace-major loop keeps each 256-centroid codebook resident in L1 cache. */
    for (uint32_t s = 0; s < m; ++s) {
        const float *sub_cents = pq->centroids + ((size_t)s * ksub * sub_dim);
        size_t col_off = (size_t)s * sub_dim;
        for (size_t i = 0; i < count; ++i) {
            const float *sub_vec = in + i * dim + col_off;
            uint32_t best = nanovec_kmeans_assign_one(sub_vec, sub_cents, ksub, sub_dim, NULL);
            out_codes[i * m + s] = (uint8_t)best;
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_decode(
    const nanovec_pq_t *pq,
    const uint8_t      *in_code,
    float              *out_vec
) {
    if (NANOVEC_UNLIKELY(pq == NULL || !pq->trained || in_code == NULL || out_vec == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_decode requires trained pq and non-null in_code/out_vec"
        );
    }

    uint32_t m       = pq->m;
    uint32_t sub_dim = pq->sub_dim;
    size_t   sub_bytes = (size_t)sub_dim * sizeof(float);

    for (uint32_t s = 0; s < m; ++s) {
        const float *c = nanovec_pq_centroid(pq, s, in_code[s]);
        memcpy(out_vec + (size_t)s * sub_dim, c, sub_bytes);
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_decode_batch(
    const nanovec_pq_t *pq,
    const uint8_t      *in_codes,
    float              *out_vecs,
    size_t              count
) {
    if (NANOVEC_UNLIKELY(
            pq == NULL || !pq->trained || (count > 0 && (in_codes == NULL || out_vecs == NULL))
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_decode_batch requires trained pq and valid buffers"
        );
    }

    uint32_t dim = pq->dim;
    uint32_t m   = pq->m;

    for (size_t i = 0; i < count; ++i) {
        nanovec_status_t st = nanovec_pq_decode(pq, in_codes + i * m, out_vecs + i * dim);
        if (NANOVEC_UNLIKELY(st != NANOVEC_SUCCESS)) {
            return st;
        }
    }
    return NANOVEC_SUCCESS;
}

size_t nanovec_pq_serialized_size(uint16_t dim) {
    if (dim == 0) {
        return 0U;
    }
    size_t payload = (size_t)NANOVEC_PQ_K * (size_t)dim * sizeof(float);
    return NANOVEC_PQ_SERIAL_HDR + nanovec_align_up(payload, NANOVEC_CACHE_LINE);
}

nanovec_status_t nanovec_pq_serialize(
    const nanovec_pq_t *pq,
    void               *out_buf,
    size_t              buf_cap,
    size_t             *out_written
) {
    if (NANOVEC_UNLIKELY(pq == NULL || !pq->trained || pq->centroids == NULL || out_buf == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_serialize requires trained pq and non-null out_buf"
        );
    }

    size_t total = nanovec_pq_serialized_size(pq->dim);
    if (NANOVEC_UNLIKELY(buf_cap < total)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "pq_serialize buffer too small (%zu < %zu)",
            buf_cap,
            total
        );
    }

    uint8_t *dst = (uint8_t *)out_buf;
    memset(dst, 0, NANOVEC_PQ_SERIAL_HDR);
    dst[0] = 'N';
    dst[1] = 'V';
    dst[2] = 'P';
    dst[3] = 'Q';
    nanovec_write_le16(dst + 0x04, (uint16_t)NANOVEC_PQ_SERIAL_VERSION);
    nanovec_write_le16(dst + 0x06, pq->dim);
    nanovec_write_le16(dst + 0x08, pq->m);
    nanovec_write_le16(dst + 0x0A, pq->sub_dim);
    nanovec_write_le16(dst + 0x0C, pq->ksub);
    nanovec_write_le16(dst + 0x0E, (uint16_t)NANOVEC_PQ_FLAG_TRAINED);

    size_t nfloats   = (size_t)pq->ksub * (size_t)pq->dim;
    size_t raw_bytes = nfloats * sizeof(float);
    uint8_t *payload = dst + NANOVEC_PQ_SERIAL_HDR;

#if NANOVEC_LITTLE_ENDIAN
    memcpy(payload, pq->centroids, raw_bytes);
#else
    for (size_t i = 0; i < nfloats; ++i) {
        nanovec_write_f32_le(payload + i * sizeof(float), pq->centroids[i]);
    }
#endif

    size_t pad = total - NANOVEC_PQ_SERIAL_HDR - raw_bytes;
    if (pad > 0U) {
        memset(payload + raw_bytes, 0, pad);
    }

    if (out_written != NULL) {
        *out_written = total;
    }
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_pq_parse_header(
    const uint8_t *src,
    size_t         buf_size,
    uint16_t      *out_dim,
    uint16_t      *out_m,
    uint16_t      *out_sub_dim,
    uint16_t      *out_ksub,
    uint16_t      *out_flags
) {
    if (NANOVEC_UNLIKELY(src == NULL || buf_size < NANOVEC_PQ_SERIAL_HDR)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "pq codebook buffer truncated (%zu bytes)",
            buf_size
        );
    }
    if (NANOVEC_UNLIKELY(src[0] != 'N' || src[1] != 'V' || src[2] != 'P' || src[3] != 'Q')) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_CORRUPT, "invalid pq codebook magic");
    }

    uint16_t ver = nanovec_read_le16(src + 0x04);
    if (NANOVEC_UNLIKELY(ver != NANOVEC_PQ_SERIAL_VERSION)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_VERSION,
            "unsupported pq codebook version %u",
            (unsigned)ver
        );
    }

    uint16_t dim     = nanovec_read_le16(src + 0x06);
    uint16_t m       = nanovec_read_le16(src + 0x08);
    uint16_t sub_dim = nanovec_read_le16(src + 0x0A);
    uint16_t ksub    = nanovec_read_le16(src + 0x0C);
    uint16_t flags   = nanovec_read_le16(src + 0x0E);

    if (NANOVEC_UNLIKELY(
            dim == 0 || m == 0 || (dim % m) != 0 || sub_dim != (uint16_t)(dim / m) ||
            ksub != NANOVEC_PQ_K || (flags & NANOVEC_PQ_FLAG_TRAINED) == 0U
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "corrupt pq codebook header (dim=%u m=%u sub=%u ksub=%u flags=0x%x)",
            (unsigned)dim, (unsigned)m, (unsigned)sub_dim, (unsigned)ksub, (unsigned)flags
        );
    }

    size_t required = nanovec_pq_serialized_size(dim);
    if (NANOVEC_UNLIKELY(buf_size < required)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "pq codebook payload truncated (%zu < %zu)",
            buf_size,
            required
        );
    }

    *out_dim     = dim;
    *out_m       = m;
    *out_sub_dim = sub_dim;
    *out_ksub    = ksub;
    *out_flags   = flags;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_deserialize(
    const void   *buf,
    size_t        buf_size,
    nanovec_pq_t *out_pq
) {
    if (NANOVEC_UNLIKELY(out_pq == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "out_pq is null");
    }

    const uint8_t *src = (const uint8_t *)buf;
    uint16_t dim = 0, m = 0, sub_dim = 0, ksub = 0, flags = 0;
    nanovec_status_t st = nanovec_pq_parse_header(src, buf_size, &dim, &m, &sub_dim, &ksub, &flags);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    (void)sub_dim;
    (void)flags;

    st = nanovec_pq_init(out_pq, dim, m);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    size_t nfloats = (size_t)ksub * (size_t)dim;
    const uint8_t *payload = src + NANOVEC_PQ_SERIAL_HDR;
    for (size_t i = 0; i < nfloats; ++i) {
        float v = nanovec_read_f32_le(payload + i * sizeof(float));
        if (NANOVEC_UNLIKELY(!isfinite(v))) {
            nanovec_pq_destroy(out_pq);
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_CORRUPT,
                "pq codebook contains non-finite centroid at index %zu",
                i
            );
        }
        out_pq->centroids[i] = v;
    }

    out_pq->trained = true;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_pq_attach_mmap(
    const void   *buf,
    size_t        buf_size,
    nanovec_pq_t *out_pq
) {
    if (NANOVEC_UNLIKELY(out_pq == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "out_pq is null");
    }

    const uint8_t *src = (const uint8_t *)buf;
    uint16_t dim = 0, m = 0, sub_dim = 0, ksub = 0, flags = 0;
    nanovec_status_t st = nanovec_pq_parse_header(src, buf_size, &dim, &m, &sub_dim, &ksub, &flags);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    (void)flags;

#if NANOVEC_LITTLE_ENDIAN
    const uint8_t *payload = src + NANOVEC_PQ_SERIAL_HDR;
    if ((((uintptr_t)payload) & (NANOVEC_CACHE_LINE - 1U)) == 0U) {
        memset(out_pq, 0, sizeof(*out_pq));
        out_pq->dim            = dim;
        out_pq->m              = m;
        out_pq->sub_dim        = sub_dim;
        out_pq->ksub           = ksub;
        out_pq->trained        = true;
        out_pq->owns_centroids = false;
        out_pq->centroids      = (float *)(uintptr_t)payload;
        return NANOVEC_SUCCESS;
    }
#endif
    return nanovec_pq_deserialize(buf, buf_size, out_pq);
}
