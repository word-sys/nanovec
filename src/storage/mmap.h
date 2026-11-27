/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_MMAP_H
#define NANOVEC_STORAGE_MMAP_H

#include "core/types.h"
#include "storage/file.h"
#include "storage/format.h"

typedef struct nanovec_mmap {
    nanovec_file_t file;
    uint8_t       *base;
    size_t         mapped_len;
    bool           readonly;
} nanovec_mmap_t;

nanovec_status_t nanovec_mmap_open(
    const char     *path,
    size_t          min_size,
    bool            readonly,
    bool            create,
    nanovec_mmap_t *out_map
);

nanovec_status_t nanovec_mmap_open_nvec(
    const char                  *path,
    const nanovec_file_header_t *init_hdr,
    size_t                       initial_file_size,
    bool                         readonly,
    bool                         create,
    nanovec_mmap_t              *out_map
);

nanovec_status_t nanovec_mmap_sync(const nanovec_mmap_t *map, bool async);
nanovec_status_t nanovec_mmap_sync_range(
    const nanovec_mmap_t *map,
    uint64_t              offset,
    size_t                len,
    bool                  async
);
nanovec_status_t nanovec_mmap_close(nanovec_mmap_t *map);

nanovec_status_t nanovec_madvise_seq(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_rand(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_dontneed(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_normal(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_willneed(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_mmap_resident_pages(
    const nanovec_mmap_t *map,
    uint64_t              offset,
    size_t                len,
    size_t               *out_resident,
    size_t               *out_total
);

/* Passing offset=0, len=0 applies the paging hint to the entire mapped region. */
nanovec_status_t nanovec_madvise_seq(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_rand(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_dontneed(const nanovec_mmap_t *map, uint64_t offset, size_t len);
nanovec_status_t nanovec_madvise_normal(const nanovec_mmap_t *map, uint64_t offset, size_t len);

static inline nanovec_file_header_t *nanovec_mmap_header(const nanovec_mmap_t *map) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL || map->mapped_len < NANOVEC_HEADER_SIZE)) {
        return NULL;
    }
    return (nanovec_file_header_t *)(void *)map->base;
}

static inline void *nanovec_mmap_ptr(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL)) {
        return NULL;
    }
    if (NANOVEC_UNLIKELY(offset > (uint64_t)map->mapped_len || len > map->mapped_len - (size_t)offset)) {
        return NULL;
    }
    return (void *)(map->base + (size_t)offset);
}

static inline void *nanovec_mmap_vectors_ptr(const nanovec_mmap_t *map) {
    const nanovec_file_header_t *hdr = nanovec_mmap_header(map);
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return NULL;
    }
    uint64_t off = nanovec_le64toh(hdr->vectors_offset);
    if (off == 0ULL || off >= (uint64_t)map->mapped_len) {
        return NULL;
    }
    return (void *)(map->base + (size_t)off);
}

static inline void *nanovec_mmap_quant_ptr(const nanovec_mmap_t *map) {
    const nanovec_file_header_t *hdr = nanovec_mmap_header(map);
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return NULL;
    }
    uint64_t off = nanovec_le64toh(hdr->quant_offset);
    if (off == 0ULL || off >= (uint64_t)map->mapped_len) {
        return NULL;
    }
    return (void *)(map->base + (size_t)off);
}

static inline void *nanovec_mmap_graph_ptr(const nanovec_mmap_t *map) {
    const nanovec_file_header_t *hdr = nanovec_mmap_header(map);
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return NULL;
    }
    uint64_t off = nanovec_le64toh(hdr->graph_offset);
    if (off == 0ULL || off >= (uint64_t)map->mapped_len) {
        return NULL;
    }
    return (void *)(map->base + (size_t)off);
}

static inline void *nanovec_mmap_metadata_ptr(const nanovec_mmap_t *map) {
    const nanovec_file_header_t *hdr = nanovec_mmap_header(map);
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return NULL;
    }
    uint64_t off = nanovec_le64toh(hdr->metadata_offset);
    if (off == 0ULL || off >= (uint64_t)map->mapped_len) {
        return NULL;
    }
    return (void *)(map->base + (size_t)off);
}

#endif /* NANOVEC_STORAGE_MMAP_H */
