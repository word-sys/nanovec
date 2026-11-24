#ifndef NANOVEC_QUANT_SQ8_H
#define NANOVEC_QUANT_SQ8_H

#include "core/types.h"

typedef struct nanovec_sq8 {
    uint16_t dim;
    bool     per_dim;
    bool     trained;
    float   *min_val;
    float   *max_val;
    float   *scale;
    float   *inv_scale;
} nanovec_sq8_t;

nanovec_status_t nanovec_sq8_init(nanovec_sq8_t *sq, uint16_t dim, bool per_dim);
void             nanovec_sq8_destroy(nanovec_sq8_t *sq);

nanovec_status_t nanovec_sq8_train(
    nanovec_sq8_t *sq,
    const float   *vectors,
    size_t         count
);

nanovec_status_t nanovec_sq8_encode(
    const nanovec_sq8_t *sq,
    const float         *in,
    uint8_t             *out
);

nanovec_status_t nanovec_sq8_encode_batch(
    const nanovec_sq8_t *sq,
    const float         *in,
    uint8_t             *out,
    size_t               count
);

#endif /* NANOVEC_QUANT_SQ8_H */
