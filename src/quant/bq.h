#ifndef NANOVEC_QUANT_BQ_H
#define NANOVEC_QUANT_BQ_H

#include "core/types.h"

static inline uint32_t nanovec_bq_num_words(uint32_t dim) {
    return (dim + 63U) >> 6;
}

static inline size_t nanovec_bq_byte_size(uint32_t dim) {
    return (size_t)nanovec_bq_num_words(dim) * sizeof(uint64_t);
}

void nanovec_bq_encode_scalar(const float *in, uint32_t dim, uint64_t *out);

nanovec_status_t nanovec_bq_encode(const float *in, uint32_t dim, uint64_t *out);
nanovec_status_t nanovec_bq_encode_batch(
    const float *in,
    uint32_t     dim,
    size_t       count,
    uint64_t    *out
);
nanovec_status_t nanovec_bq_decode(const uint64_t *in, uint32_t dim, float *out);

uint32_t nanovec_bq_distance_scalar(const uint64_t *a, const uint64_t *b, uint32_t num_words);
uint32_t nanovec_bq_distance(const uint64_t *a, const uint64_t *b, uint32_t num_words);

#endif /* NANOVEC_QUANT_BQ_H */
