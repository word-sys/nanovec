/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "quant/kmeans.h"
#include "core/alloc.h"
#include "core/error.h"
#include "math/distance.h"
#include "math/prefetch.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define NANOVEC_KMEANS_DEFAULT_ITERS 25U
#define NANOVEC_KMEANS_DEFAULT_TOL   1e-6f
#define NANOVEC_KMEANS_DEFAULT_SEED  0x4B4D45414E530001ULL
#define NANOVEC_KMEANS_BLOCK_SIZE    64U

static inline uint64_t nanovec_km_rng_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double nanovec_km_rng_f64(uint64_t *state) {
    return (double)(nanovec_km_rng_u64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static void nanovec_km_normalize_vec(float *v, uint32_t dim) {
    double sq = 0.0;
    for (uint32_t d = 0; d < dim; ++d) {
        sq += (double)v[d] * (double)v[d];
    }
    if (sq > 0.0) {
        float inv = (float)(1.0 / sqrt(sq));
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] *= inv;
        }
    }
}

uint32_t nanovec_kmeans_assign_one(
    const float *vec,
    const float *centroids,
    uint32_t     k,
    uint32_t     dim,
    float       *out_min_dist
) {
    nanovec_dist_f32_fn_t l2 = nanovec_dist_l2_fn;
    uint32_t best_c = 0;
    float    best_d = FLT_MAX;

    for (uint32_t c = 0; c < k; ++c) {
        if (c + 1U < k) {
            nanovec_prefetch_l1(centroids + (size_t)(c + 1U) * dim);
        }
        float d = l2(vec, centroids + (size_t)c * dim, dim);
        if (d < best_d) {
            best_d = d;
            best_c = c;
        }
    }

    if (out_min_dist != NULL) {
        *out_min_dist = best_d;
    }
    return best_c;
}

static void nanovec_kmeans_pp_init(
    const float *data,
    size_t       n,
    uint32_t     dim,
    uint32_t     k,
    bool         spherical,
    uint64_t    *rng,
    float       *centroids,
    double      *min_d2
) {
    nanovec_dist_f32_fn_t l2 = nanovec_dist_l2_fn;

    size_t first = (size_t)(nanovec_km_rng_u64(rng) % (uint64_t)n);
    memcpy(centroids, data + first * dim, (size_t)dim * sizeof(float));
    if (spherical) {
        nanovec_km_normalize_vec(centroids, dim);
    }

    double total_d2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (i + 1U < n) {
            nanovec_prefetch_l1(data + (i + 1U) * dim);
        }
        double d2 = (double)l2(data + i * dim, centroids, dim);
        min_d2[i] = d2;
        total_d2 += d2;
    }

    for (uint32_t c = 1U; c < k; ++c) {
        size_t chosen = 0;
        if (total_d2 <= 0.0) {
            chosen = (size_t)(nanovec_km_rng_u64(rng) % (uint64_t)n);
        } else {
            double target = nanovec_km_rng_f64(rng) * total_d2;
            double accum  = 0.0;
            chosen = n - 1U;
            for (size_t i = 0; i < n; ++i) {
                accum += min_d2[i];
                if (accum >= target) {
                    chosen = i;
                    break;
                }
            }
        }

        float *dst = centroids + (size_t)c * dim;
        memcpy(dst, data + chosen * dim, (size_t)dim * sizeof(float));
        if (spherical) {
            nanovec_km_normalize_vec(dst, dim);
        }

        if (c + 1U < k) {
            double next_total = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (i + 1U < n) {
                    nanovec_prefetch_l1(data + (i + 1U) * dim);
                }
                double d2 = (double)l2(data + i * dim, dst, dim);
                if (d2 < min_d2[i]) {
                    min_d2[i] = d2;
                }
                next_total += min_d2[i];
            }
            total_d2 = next_total;
        }
    }
}

nanovec_status_t nanovec_kmeans_fit(
    const float                   *data,
    size_t                         n,
    uint32_t                       dim,
    const nanovec_kmeans_params_t *params,
    float                         *out_centroids,
    uint32_t                      *out_assignments,
    nanovec_kmeans_result_t       *out_result
) {
    if (NANOVEC_UNLIKELY(
            data == NULL || out_centroids == NULL || params == NULL ||
            n == 0 || dim == 0 || params->k == 0
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "kmeans_fit requires non-null data/centroids/params and n,dim,k > 0"
        );
    }

    uint32_t k         = params->k;
    uint32_t max_iters = params->max_iters > 0U ? params->max_iters : NANOVEC_KMEANS_DEFAULT_ITERS;
    float    tol       = params->tol > 0.0f ? params->tol : NANOVEC_KMEANS_DEFAULT_TOL;
    uint64_t rng       = params->seed != 0ULL ? params->seed : NANOVEC_KMEANS_DEFAULT_SEED;
    bool     spherical = params->spherical;

    if (n <= (size_t)k) {
        for (uint32_t c = 0; c < k; ++c) {
            size_t src_idx = (size_t)c % n;
            float *dst = out_centroids + (size_t)c * dim;
            memcpy(dst, data + src_idx * dim, (size_t)dim * sizeof(float));
            if (spherical) {
                nanovec_km_normalize_vec(dst, dim);
            }
        }
        if (out_assignments != NULL) {
            for (size_t i = 0; i < n; ++i) {
                out_assignments[i] = (uint32_t)i;
            }
        }
        if (out_result != NULL) {
            out_result->iters_run = 0U;
            out_result->inertia   = 0.0f;
            out_result->converged = true;
        }
        return NANOVEC_SUCCESS;
    }

    size_t kd = (size_t)k * (size_t)dim;
    double   *point_d2  = (double   *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, n * sizeof(double));
    double   *sums      = (double   *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, kd * sizeof(double));
    uint64_t *counts    = (uint64_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, (size_t)k * sizeof(uint64_t));
    uint32_t *assign    = out_assignments != NULL
                              ? out_assignments
                              : (uint32_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, n * sizeof(uint32_t));
    float    *centroids = (float    *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, kd * sizeof(float));

    if (NANOVEC_UNLIKELY(!point_d2 || !sums || !counts || !assign || !centroids)) {
        nanovec_aligned_free(point_d2);
        nanovec_aligned_free(sums);
        nanovec_aligned_free(counts);
        if (out_assignments == NULL) nanovec_aligned_free(assign);
        nanovec_aligned_free(centroids);
        return NANOVEC_ERR_OOM;
    }

    nanovec_kmeans_pp_init(data, n, dim, k, spherical, &rng, centroids, point_d2);
    for (size_t i = 0; i < n; ++i) {
        assign[i] = UINT32_MAX;
    }

    nanovec_dist_f32_fn_t l2 = nanovec_dist_l2_fn;
    uint32_t iters_run = 0U;
    double   inertia   = 0.0;
    bool     converged = false;

    for (uint32_t iter = 0; iter < max_iters; ++iter) {
        iters_run = iter + 1U;
        memset(sums, 0, kd * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(uint64_t));

        size_t changed = 0U;
        inertia = 0.0;

        for (size_t blk = 0; blk < n; blk += NANOVEC_KMEANS_BLOCK_SIZE) {
            size_t blk_end = blk + NANOVEC_KMEANS_BLOCK_SIZE;
            if (blk_end > n) {
                blk_end = n;
            }
            for (size_t i = blk; i < blk_end; ++i) {
                const float *v = data + i * dim;
                if (i + 1U < blk_end) {
                    nanovec_prefetch_l1(data + (i + 1U) * dim);
                }

                float    best_d = 0.0f;
                uint32_t best_c = nanovec_kmeans_assign_one(v, centroids, k, dim, &best_d);

                if (assign[i] != best_c) {
                    assign[i] = best_c;
                    changed++;
                }
                point_d2[i] = (double)best_d;
                inertia    += (double)best_d;

                counts[best_c]++;
                double *acc = sums + (size_t)best_c * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    acc[d] += (double)v[d];
                }
            }
        }

        float max_shift_sq = 0.0f;
        for (uint32_t c = 0; c < k; ++c) {
            float *cent = centroids + (size_t)c * dim;
            if (counts[c] == 0ULL) {
                /* Re-seed empty cluster to the current highest-distortion point. */
                size_t far_idx = 0;
                double far_d2  = -1.0;
                for (size_t i = 0; i < n; ++i) {
                    if (point_d2[i] > far_d2) {
                        far_d2  = point_d2[i];
                        far_idx = i;
                    }
                }
                point_d2[far_idx] = 0.0;
                const float *src = data + far_idx * dim;
                float shift = l2(cent, src, dim);
                if (shift > max_shift_sq) {
                    max_shift_sq = shift;
                }
                memcpy(cent, src, (size_t)dim * sizeof(float));
                if (spherical) {
                    nanovec_km_normalize_vec(cent, dim);
                }
                continue;
            }

            double *acc   = sums + (size_t)c * dim;
            double  scale = 1.0 / (double)counts[c];
            if (spherical) {
                double nsq = 0.0;
                for (uint32_t d = 0; d < dim; ++d) {
                    nsq += acc[d] * acc[d];
                }
                if (nsq > 0.0) {
                    scale = 1.0 / sqrt(nsq);
                }
            }
            double shift = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                float  next_v = (float)(acc[d] * scale);
                double diff   = (double)next_v - (double)cent[d];
                shift += diff * diff;
                cent[d] = next_v;
            }
            if ((float)shift > max_shift_sq) {
                max_shift_sq = (float)shift;
            }
        }

        if (changed == 0U || max_shift_sq <= tol) {
            converged = true;
            break;
        }
    }

    memcpy(out_centroids, centroids, kd * sizeof(float));

    if (out_result != NULL) {
        out_result->iters_run = iters_run;
        out_result->inertia   = (float)inertia;
        out_result->converged = converged;
    }

    nanovec_aligned_free(point_d2);
    nanovec_aligned_free(sums);
    nanovec_aligned_free(counts);
    if (out_assignments == NULL) {
        nanovec_aligned_free(assign);
    }
    nanovec_aligned_free(centroids);
    return NANOVEC_SUCCESS;
}
