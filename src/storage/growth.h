/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_GROWTH_H
#define NANOVEC_STORAGE_GROWTH_H

#include "core/types.h"
#include "storage/format.h"
#include "storage/mmap.h"

#include <pthread.h>
#include <stdatomic.h>

typedef struct nanovec_growth_ptrs {
    uint8_t               *base;
    size_t                 mapped_len;
    nanovec_file_header_t *header;
    void                  *vectors;
    void                  *quant;
    void                  *graph;
    void                  *metadata;
    uint64_t               generation;
} nanovec_growth_ptrs_t;

typedef struct nanovec_growable_map {
    nanovec_mmap_t        map;
    nanovec_growth_ptrs_t ptrs;
    pthread_rwlock_t      rwlock;
    bool                  lock_init;
    _Atomic uint64_t      generation;
} nanovec_growable_map_t;

size_t nanovec_growth_calc_size(size_t current_size, size_t required_size, size_t page_size);

nanovec_status_t nanovec_mmap_grow(nanovec_mmap_t *map, size_t new_size, bool *out_moved);

nanovec_status_t nanovec_growth_init(nanovec_growable_map_t *gmap, nanovec_mmap_t map);
nanovec_status_t nanovec_growth_destroy(nanovec_growable_map_t *gmap);

nanovec_status_t nanovec_growth_read_lock(
    nanovec_growable_map_t *gmap,
    nanovec_growth_ptrs_t  *out_ptrs
);
void nanovec_growth_read_unlock(nanovec_growable_map_t *gmap);

nanovec_status_t nanovec_growth_ensure_capacity(
    nanovec_growable_map_t *gmap,
    size_t                  required_bytes,
    bool                   *out_grew
);

nanovec_status_t nanovec_growth_grow_exact(
    nanovec_growable_map_t *gmap,
    size_t                  new_size,
    bool                   *out_moved
);

nanovec_status_t nanovec_growth_refresh_ptrs(nanovec_growable_map_t *gmap);

#endif /* NANOVEC_STORAGE_GROWTH_H */
