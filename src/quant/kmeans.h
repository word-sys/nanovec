/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_QUANT_KMEANS_H
#define NANOVEC_QUANT_KMEANS_H

#include "core/types.h"

typedef struct nanovec_kmeans_params {
    uint32_t k;
    uint32_t max_iters;
    float    tol;
    uint64_t seed;
    bool     spherical;
} nanovec_kmeans_params_t;

typedef struct nanovec_kmeans_result {
    uint32_t iters_run;
    float    inertia;
    bool     converged;
} nanovec_kmeans_result_t;

uint32_t nanovec_kmeans_assign_one(
    const float *vec,
    const float *centroids,
    uint32_t     k,
    uint32_t     dim,
    float       *out_min_dist
);

nanovec_status_t nanovec_kmeans_fit(
    const float                   *data,
    size_t                         n,
    uint32_t                       dim,
    const nanovec_kmeans_params_t *params,
    float                         *out_centroids,
    uint32_t                      *out_assignments,
    nanovec_kmeans_result_t       *out_result
);

#endif /* NANOVEC_QUANT_KMEANS_H */
