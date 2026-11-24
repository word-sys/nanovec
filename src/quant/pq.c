#include "quant/pq.h"
#include "core/alloc.h"
#include "core/error.h"
#include "quant/kmeans.h"

#include <string.h>

#define NANOVEC_PQ_DEFAULT_SEED 0x5051545241494E01ULL

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

    pq->dim       = dim;
    pq->m         = m;
    pq->sub_dim   = (uint16_t)(dim / m);
    pq->ksub      = (uint16_t)NANOVEC_PQ_K;
    pq->trained   = false;
    pq->centroids = centroids;
    return NANOVEC_SUCCESS;
}

void nanovec_pq_destroy(nanovec_pq_t *pq) {
    if (pq == NULL) {
        return;
    }
    nanovec_aligned_free(pq->centroids);
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
