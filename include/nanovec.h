/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_H
#define NANOVEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "nanovec_version.h"

#if defined(__GNUC__) || defined(__clang__)
#define NANOVEC_API __attribute__((visibility("default")))
#else
#define NANOVEC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nanovec_db nanovec_db_t;
typedef struct nanovec_index nanovec_index_t;
typedef struct nanovec_query nanovec_query_t;
typedef struct nanovec_model nanovec_model_t;

typedef enum nanovec_status {
    NANOVEC_SUCCESS            = 0,
    NANOVEC_ERR_INVALID_ARG    = -1,
    NANOVEC_ERR_OOM            = -2,
    NANOVEC_ERR_IO             = -3,
    NANOVEC_ERR_CORRUPT        = -4,
    NANOVEC_ERR_NOT_FOUND      = -5,
    NANOVEC_ERR_ALREADY_EXISTS = -6,
    NANOVEC_ERR_FULL           = -7,
    NANOVEC_ERR_DIM_MISMATCH   = -8,
    NANOVEC_ERR_UNSUPPORTED    = -9,
    NANOVEC_ERR_MMAP           = -10,
    NANOVEC_ERR_WAL            = -11,
    NANOVEC_ERR_MODEL          = -12,
    NANOVEC_ERR_TOKENIZER      = -13,
    NANOVEC_ERR_INTERNAL       = -14,
    NANOVEC_ERR_BUSY           = -15,
    NANOVEC_ERR_VERSION        = -16
} nanovec_status_t;

/* Values match on-disk .nvec header fields 1:1. */
typedef enum nanovec_metric {
    NANOVEC_METRIC_L2      = 0,
    NANOVEC_METRIC_COSINE  = 1,
    NANOVEC_METRIC_DOT     = 2,
    NANOVEC_METRIC_HAMMING = 3
} nanovec_metric_t;

typedef enum nanovec_index_type {
    NANOVEC_INDEX_FLAT   = 0,
    NANOVEC_INDEX_IVF    = 1,
    NANOVEC_INDEX_HNSW   = 2,
    NANOVEC_INDEX_VAMANA = 3,
    NANOVEC_INDEX_AUTO   = 255
} nanovec_index_type_t;

typedef enum nanovec_quant {
    NANOVEC_QUANT_NONE = 0,
    NANOVEC_QUANT_SQ8  = 1,
    NANOVEC_QUANT_BQ   = 2,
    NANOVEC_QUANT_PQ   = 3
} nanovec_quant_t;

#define NANOVEC_OPEN_CREATE      (1U << 0)
#define NANOVEC_OPEN_READONLY    (1U << 1)
#define NANOVEC_OPEN_WAL         (1U << 2)
#define NANOVEC_OPEN_DIRECT_IO   (1U << 3)
#define NANOVEC_OPEN_MMAP_RANDOM (1U << 4)

#define NANOVEC_SEARCH_EXACT     (1U << 0)

typedef struct nanovec_result {
    uint64_t id;
    float    distance;
    uint32_t _pad;
} nanovec_result_t;

typedef struct nanovec_config {
    uint16_t    dim;
    uint16_t    metric;
    uint16_t    index_type;
    uint16_t    quant;
    uint64_t    initial_capacity;
    uint32_t    m;
    uint32_t    ef_construction;
    uint32_t    ef_search;
    uint32_t    ivf_nlist;
    uint32_t    ivf_nprobe;
    uint32_t    pq_subspaces;
    uint32_t    num_threads;
    float       vamana_alpha;
    uint32_t    flags;
    const char *model_path;
} nanovec_config_t;

typedef struct nanovec_search_params {
    uint32_t top_k;
    uint32_t ef_search;
    uint32_t nprobe;
    uint32_t rerank_factor;
    uint32_t flags;
} nanovec_search_params_t;

typedef struct nanovec_stats {
    uint64_t count;
    uint64_t capacity;
    uint64_t deleted_count;
    uint64_t file_size_bytes;
    uint64_t memory_bytes;
    uint16_t dim;
    uint16_t metric;
    uint16_t index_type;
    uint16_t quant;
    uint32_t simd_caps;
    uint32_t _pad;
} nanovec_stats_t;

NANOVEC_API const char *nanovec_version_string(void);
NANOVEC_API int         nanovec_version_number(void);
NANOVEC_API const char *nanovec_status_string(nanovec_status_t status);
NANOVEC_API const char *nanovec_get_last_error(void);
NANOVEC_API void        nanovec_clear_error(void);

NANOVEC_API void nanovec_config_default(nanovec_config_t *cfg);
NANOVEC_API void nanovec_search_params_default(nanovec_search_params_t *params);

NANOVEC_API nanovec_status_t nanovec_db_open(
    const char             *path,
    const nanovec_config_t *cfg,
    nanovec_db_t          **out_db
);
NANOVEC_API nanovec_status_t nanovec_db_close(nanovec_db_t *db);
NANOVEC_API nanovec_status_t nanovec_db_sync(nanovec_db_t *db);
NANOVEC_API nanovec_status_t nanovec_db_compact(nanovec_db_t *db);
NANOVEC_API nanovec_status_t nanovec_db_stats(
    const nanovec_db_t *db,
    nanovec_stats_t    *out_stats
);
NANOVEC_API nanovec_index_t *nanovec_db_get_index(nanovec_db_t *db);

NANOVEC_API nanovec_status_t nanovec_db_insert(
    nanovec_db_t *db,
    uint64_t      id,
    const float  *vec,
    const void   *metadata,
    size_t        metadata_len
);
NANOVEC_API nanovec_status_t nanovec_db_insert_batch(
    nanovec_db_t   *db,
    const uint64_t *ids,
    const float    *vecs,
    size_t          count
);
NANOVEC_API nanovec_status_t nanovec_db_delete(nanovec_db_t *db, uint64_t id);
NANOVEC_API nanovec_status_t nanovec_db_get_vector(
    const nanovec_db_t *db,
    uint64_t            id,
    float              *out_vec,
    size_t              dim
);
NANOVEC_API nanovec_status_t nanovec_db_get_metadata(
    const nanovec_db_t *db,
    uint64_t            id,
    void               *out_buf,
    size_t              buf_cap,
    size_t             *out_len
);
NANOVEC_API nanovec_status_t nanovec_db_train(nanovec_db_t *db);

NANOVEC_API nanovec_status_t nanovec_db_search(
    const nanovec_db_t            *db,
    const float                   *query_vec,
    const nanovec_search_params_t *params,
    nanovec_result_t              *out_results,
    size_t                        *out_count
);
NANOVEC_API nanovec_status_t nanovec_db_search_batch(
    const nanovec_db_t            *db,
    const float                   *query_vecs,
    size_t                         num_queries,
    const nanovec_search_params_t *params,
    nanovec_result_t              *out_results,
    size_t                        *out_counts
);

NANOVEC_API nanovec_status_t nanovec_query_create(
    const nanovec_db_t            *db,
    const nanovec_search_params_t *params,
    nanovec_query_t              **out_query
);
NANOVEC_API nanovec_status_t nanovec_query_set_vector(
    nanovec_query_t *query,
    const float     *vec,
    uint16_t         dim
);
NANOVEC_API nanovec_status_t nanovec_query_set_text(
    nanovec_query_t *query,
    const char      *text
);
NANOVEC_API nanovec_status_t nanovec_query_execute(
    nanovec_query_t  *query,
    nanovec_result_t *out_results,
    size_t           *out_count
);
NANOVEC_API void nanovec_query_destroy(nanovec_query_t *query);

NANOVEC_API nanovec_status_t nanovec_index_create(
    const nanovec_config_t *cfg,
    nanovec_index_t       **out_index
);
NANOVEC_API void nanovec_index_destroy(nanovec_index_t *index);
NANOVEC_API nanovec_status_t nanovec_index_add(
    nanovec_index_t *index,
    uint64_t         id,
    const float     *vec
);
NANOVEC_API nanovec_status_t nanovec_index_remove(
    nanovec_index_t *index,
    uint64_t         id
);
NANOVEC_API nanovec_status_t nanovec_index_search(
    const nanovec_index_t         *index,
    const float                   *query_vec,
    const nanovec_search_params_t *params,
    nanovec_result_t              *out_results,
    size_t                        *out_count
);

NANOVEC_API nanovec_status_t nanovec_model_load(
    const char       *gguf_path,
    nanovec_model_t **out_model
);
NANOVEC_API void     nanovec_model_unload(nanovec_model_t *model);
NANOVEC_API uint16_t nanovec_model_dim(const nanovec_model_t *model);
NANOVEC_API nanovec_status_t nanovec_model_embed(
    const nanovec_model_t *model,
    const char            *text,
    float                 *out_vec,
    size_t                 dim
);

NANOVEC_API nanovec_status_t nanovec_db_attach_model(
    nanovec_db_t    *db,
    nanovec_model_t *model
);
NANOVEC_API nanovec_status_t nanovec_db_insert_text(
    nanovec_db_t *db,
    uint64_t      id,
    const char   *text,
    const void   *metadata,
    size_t        metadata_len
);
NANOVEC_API nanovec_status_t nanovec_db_search_text(
    const nanovec_db_t            *db,
    const char                    *text,
    const nanovec_search_params_t *params,
    nanovec_result_t              *out_results,
    size_t                        *out_count
);

#ifdef __cplusplus
}
#endif

#endif /* NANOVEC_H */
