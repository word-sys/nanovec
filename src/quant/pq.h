#ifndef NANOVEC_QUANT_PQ_H
#define NANOVEC_QUANT_PQ_H

#include "core/types.h"

#define NANOVEC_PQ_K 256U

typedef struct nanovec_pq {
    uint16_t dim;
    uint16_t m;
    uint16_t sub_dim;
    uint16_t ksub;
    bool     trained;
    float   *centroids; /* [m][256][sub_dim], 64-byte aligned */
} nanovec_pq_t;

static inline const float *nanovec_pq_centroid(
    const nanovec_pq_t *pq,
    uint32_t            subspace,
    uint32_t            code
) {
    return pq->centroids + (((size_t)subspace * pq->ksub + code) * pq->sub_dim);
}

static inline size_t nanovec_pq_codebook_floats(const nanovec_pq_t *pq) {
    return pq ? (size_t)pq->ksub * (size_t)pq->dim : 0U;
}

static inline size_t nanovec_pq_code_size(const nanovec_pq_t *pq) {
    return pq ? (size_t)pq->m : 0U;
}

nanovec_status_t nanovec_pq_init(nanovec_pq_t *pq, uint16_t dim, uint16_t m);
void             nanovec_pq_destroy(nanovec_pq_t *pq);

nanovec_status_t nanovec_pq_train(
    nanovec_pq_t *pq,
    const float  *vectors,
    size_t        count,
    uint32_t      max_iters,
    uint64_t      seed
);

nanovec_status_t nanovec_pq_encode(
    const nanovec_pq_t *pq,
    const float        *in,
    uint8_t            *out_code
);

nanovec_status_t nanovec_pq_encode_batch(
    const nanovec_pq_t *pq,
    const float        *in,
    uint8_t            *out_codes,
    size_t              count
);

nanovec_status_t nanovec_pq_decode(
    const nanovec_pq_t *pq,
    const uint8_t      *in_code,
    float              *out_vec
);

nanovec_status_t nanovec_pq_decode_batch(
    const nanovec_pq_t *pq,
    const uint8_t      *in_codes,
    float              *out_vecs,
    size_t              count
);

#endif /* NANOVEC_QUANT_PQ_H */
