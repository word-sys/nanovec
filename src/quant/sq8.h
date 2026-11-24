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

nanovec_status_t nanovec_sq8_decode(
    const nanovec_sq8_t *sq,
    const uint8_t       *in,
    float               *out
);

nanovec_status_t nanovec_sq8_decode_batch(
    const nanovec_sq8_t *sq,
    const uint8_t       *in,
    float               *out,
    size_t               count
);

typedef struct nanovec_sq8_query {
    uint16_t dim;
    int8_t  *wq;
    float    base_dot;
    float    alpha;
    int32_t  wq_sum;
} nanovec_sq8_query_t;

nanovec_status_t nanovec_sq8_prepare_query(
    const nanovec_sq8_t *sq,
    const float         *query,
    nanovec_sq8_query_t *out_q
);
void nanovec_sq8_query_destroy(nanovec_sq8_query_t *q);

int32_t nanovec_sq8_idot_scalar(const uint8_t *code, const int8_t *wq, uint32_t dim);
int32_t nanovec_sq8_idot(const uint8_t *code, const int8_t *wq, uint32_t dim);

float nanovec_sq8_dot_prepared(const nanovec_sq8_query_t *q, const uint8_t *code);
float nanovec_sq8_dist_dot_prepared(const nanovec_sq8_query_t *q, const uint8_t *code);
float nanovec_sq8_dist_dot(const nanovec_sq8_t *sq, const float *query, const uint8_t *code);
float nanovec_sq8_dist_l2(const nanovec_sq8_t *sq, const float *query, const uint8_t *code);

#endif /* NANOVEC_QUANT_SQ8_H */
