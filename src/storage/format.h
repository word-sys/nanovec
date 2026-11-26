/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_FORMAT_H
#define NANOVEC_STORAGE_FORMAT_H

#include "core/types.h"

#define NANOVEC_FILE_MAGIC_0 0x4EU /* 'N' */
#define NANOVEC_FILE_MAGIC_1 0x56U /* 'V' */
#define NANOVEC_FILE_MAGIC_2 0x45U /* 'E' */
#define NANOVEC_FILE_MAGIC_3 0x43U /* 'C' */

#define NANOVEC_FORMAT_VERSION        1U
#define NANOVEC_HEADER_SIZE           128U
#define NANOVEC_HEADER_CRC_RANGE_LEN  0x48U

#define NANOVEC_HDR_FLAG_LITTLE_ENDIAN (1U << 0)
#define NANOVEC_HDR_FLAG_HAS_WAL       (1U << 1)
#define NANOVEC_HDR_FLAG_HAS_METADATA  (1U << 2)
#define NANOVEC_HDR_FLAG_DIRTY         (1U << 3)
#define NANOVEC_HDR_FLAG_CHECKPOINT_B  (1U << 4)

typedef struct NANOVEC_PACKED nanovec_file_header {
    uint8_t  magic[4];        /* 0x00 - 0x03: "NVEC" */
    uint16_t version;         /* 0x04 - 0x05: format version (1) */
    uint16_t flags;           /* 0x06 - 0x07: 0x0001 = Little-Endian */
    uint16_t metric;          /* 0x08 - 0x09: 0=L2, 1=Cosine, 2=Dot, 3=Hamming */
    uint16_t index_type;      /* 0x0A - 0x0B: 0=Flat, 1=IVF, 2=HNSW, 3=Vamana */
    uint16_t quant;           /* 0x0C - 0x0D: 0=FP32, 1=SQ8, 2=BQ, 3=PQ */
    uint16_t dim;             /* 0x0E - 0x0F: vector dimension */
    uint64_t vector_count;    /* 0x10 - 0x17: active vector count */
    uint64_t max_capacity;    /* 0x18 - 0x1F: allocated vector slots */
    uint64_t vectors_offset;  /* 0x20 - 0x27: vector data offset (64B aligned) */
    uint64_t quant_offset;    /* 0x28 - 0x2F: quantization tables offset */
    uint64_t graph_offset;    /* 0x30 - 0x37: graph / inverted lists offset */
    uint64_t metadata_offset; /* 0x38 - 0x3F: metadata payload offset */
    uint64_t wal_offset;      /* 0x40 - 0x47: WAL / transaction offset */
    uint32_t header_crc32;    /* 0x48 - 0x4B: CRC-32 over bytes 0x00..0x47 */
    uint32_t crc32_pad;       /* 0x4C - 0x4F: zero padding */
    uint8_t  reserved[48];    /* 0x50 - 0x7F: reserved zero padding */
} nanovec_file_header_t;

NANOVEC_STATIC_ASSERT(sizeof(nanovec_file_header_t) == 128U, "header must be 128 bytes");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, magic)           == 0x00U, "magic offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, version)         == 0x04U, "version offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, flags)           == 0x06U, "flags offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, metric)          == 0x08U, "metric offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, index_type)      == 0x0AU, "index_type offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, quant)           == 0x0CU, "quant offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, dim)             == 0x0EU, "dim offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, vector_count)    == 0x10U, "vector_count offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, max_capacity)    == 0x18U, "max_capacity offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, vectors_offset)  == 0x20U, "vectors_offset offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, quant_offset)    == 0x28U, "quant_offset offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, graph_offset)    == 0x30U, "graph_offset offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, metadata_offset) == 0x38U, "metadata_offset offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, wal_offset)      == 0x40U, "wal_offset offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, header_crc32)    == 0x48U, "header_crc32 offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, crc32_pad)       == 0x4CU, "crc32_pad offset");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_file_header_t, reserved)        == 0x50U, "reserved offset");

static inline void nanovec_header_init(
    nanovec_file_header_t *hdr,
    uint16_t               dim,
    nanovec_metric_t       metric,
    nanovec_index_type_t   index_type,
    nanovec_quant_t        quant,
    uint64_t               max_capacity
) {
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic[0]     = NANOVEC_FILE_MAGIC_0;
    hdr->magic[1]     = NANOVEC_FILE_MAGIC_1;
    hdr->magic[2]     = NANOVEC_FILE_MAGIC_2;
    hdr->magic[3]     = NANOVEC_FILE_MAGIC_3;
    hdr->version      = nanovec_htole16((uint16_t)NANOVEC_FORMAT_VERSION);
    hdr->flags        = nanovec_htole16((uint16_t)NANOVEC_HDR_FLAG_LITTLE_ENDIAN);
    hdr->metric       = nanovec_htole16((uint16_t)metric);
    hdr->index_type   = nanovec_htole16((uint16_t)index_type);
    hdr->quant        = nanovec_htole16((uint16_t)quant);
    hdr->dim          = nanovec_htole16(dim);
    hdr->vector_count = nanovec_htole64(0ULL);
    hdr->max_capacity = nanovec_htole64(max_capacity);
}

static inline bool nanovec_header_magic_valid(const nanovec_file_header_t *hdr) {
    return hdr != NULL
        && hdr->magic[0] == NANOVEC_FILE_MAGIC_0
        && hdr->magic[1] == NANOVEC_FILE_MAGIC_1
        && hdr->magic[2] == NANOVEC_FILE_MAGIC_2
        && hdr->magic[3] == NANOVEC_FILE_MAGIC_3;
}

static inline nanovec_status_t nanovec_header_validate_basic(const nanovec_file_header_t *hdr) {
    if (!nanovec_header_magic_valid(hdr)) {
        return NANOVEC_ERR_CORRUPT;
    }
    uint16_t ver = nanovec_le16toh(hdr->version);
    if (ver != NANOVEC_FORMAT_VERSION) {
        return NANOVEC_ERR_VERSION;
    }
    uint16_t flags = nanovec_le16toh(hdr->flags);
    if ((flags & NANOVEC_HDR_FLAG_LITTLE_ENDIAN) == 0U) {
        return NANOVEC_ERR_UNSUPPORTED;
    }
    uint16_t metric = nanovec_le16toh(hdr->metric);
    uint16_t idx    = nanovec_le16toh(hdr->index_type);
    uint16_t quant  = nanovec_le16toh(hdr->quant);
    uint16_t dim    = nanovec_le16toh(hdr->dim);
    if (metric > NANOVEC_METRIC_HAMMING || idx > NANOVEC_INDEX_VAMANA ||
        quant > NANOVEC_QUANT_PQ || dim == 0U) {
        return NANOVEC_ERR_CORRUPT;
    }
    uint64_t count = nanovec_le64toh(hdr->vector_count);
    uint64_t cap   = nanovec_le64toh(hdr->max_capacity);
    if (count > cap) {
        return NANOVEC_ERR_CORRUPT;
    }
    uint64_t vec_off   = nanovec_le64toh(hdr->vectors_offset);
    uint64_t quant_off = nanovec_le64toh(hdr->quant_offset);
    uint64_t graph_off = nanovec_le64toh(hdr->graph_offset);
    if ((vec_off & (NANOVEC_CACHE_LINE - 1U)) != 0ULL ||
        (quant_off & (NANOVEC_CACHE_LINE - 1U)) != 0ULL ||
        (graph_off & (NANOVEC_CACHE_LINE - 1U)) != 0ULL) {
        return NANOVEC_ERR_CORRUPT;
    }
    return NANOVEC_SUCCESS;
}

static inline size_t nanovec_vector_stride_bytes(
    uint16_t        dim,
    nanovec_quant_t quant,
    uint16_t        pq_m
) {
    switch (quant) {
        case NANOVEC_QUANT_NONE: return (size_t)dim * sizeof(float);
        case NANOVEC_QUANT_SQ8:  return (size_t)dim;
        case NANOVEC_QUANT_BQ:   return (size_t)((dim + 63U) >> 6) * sizeof(uint64_t);
        case NANOVEC_QUANT_PQ:   return (size_t)pq_m;
    }
    return 0U;
}

#endif /* NANOVEC_STORAGE_FORMAT_H */
