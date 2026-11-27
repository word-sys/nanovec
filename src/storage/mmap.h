/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_MMAP_H
#define NANOVEC_STORAGE_MMAP_H

#include "core/types.h"
#include "storage/file.h"
#include "storage/format.h"

#include <pthread.h>
#include <stdatomic.h>

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

typedef struct nanovec_seqlock {
    _Alignas(NANOVEC_CACHE_LINE) _Atomic uint64_t seq;
} nanovec_seqlock_t;

static inline void nanovec_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

static inline void nanovec_seqlock_init(nanovec_seqlock_t *sl) {
    atomic_init(&sl->seq, 0ULL);
}

static inline void nanovec_seqlock_write_begin(nanovec_seqlock_t *sl) {
    uint64_t s = atomic_load_explicit(&sl->seq, memory_order_relaxed);
    atomic_store_explicit(&sl->seq, s + 1ULL, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
}

static inline void nanovec_seqlock_write_end(nanovec_seqlock_t *sl) {
    uint64_t s = atomic_load_explicit(&sl->seq, memory_order_relaxed);
    atomic_store_explicit(&sl->seq, s + 1ULL, memory_order_release);
}

static inline uint64_t nanovec_seqlock_read_begin(const nanovec_seqlock_t *sl) {
    for (;;) {
        uint64_t s = atomic_load_explicit(&sl->seq, memory_order_acquire);
        if (NANOVEC_LIKELY((s & 1ULL) == 0ULL)) {
            return s;
        }
        nanovec_cpu_relax();
    }
}

static inline bool nanovec_seqlock_read_retry(const nanovec_seqlock_t *sl, uint64_t start_seq) {
    atomic_thread_fence(memory_order_acquire);
    uint64_t end_seq = atomic_load_explicit(&sl->seq, memory_order_relaxed);
    return NANOVEC_UNLIKELY(end_seq != start_seq);
}

typedef struct nanovec_header_snapshot {
    uint64_t vector_count;
    uint64_t max_capacity;
    uint64_t vectors_offset;
    uint64_t quant_offset;
    uint64_t graph_offset;
    uint64_t metadata_offset;
    uint64_t sequence;
} nanovec_header_snapshot_t;

typedef struct nanovec_mmap_header_state {
    nanovec_seqlock_t seqlock;
    _Atomic uint64_t  vector_count;
    _Atomic uint64_t  max_capacity;
    _Atomic uint64_t  vectors_offset;
    _Atomic uint64_t  quant_offset;
    _Atomic uint64_t  graph_offset;
    _Atomic uint64_t  metadata_offset;
    pthread_mutex_t   write_lock;
    bool              lock_init;
} nanovec_mmap_header_state_t;

nanovec_status_t nanovec_mmap_header_state_init(
    nanovec_mmap_header_state_t *state,
    const nanovec_mmap_t        *map
);
void nanovec_mmap_header_state_destroy(nanovec_mmap_header_state_t *state);

nanovec_status_t nanovec_mmap_header_read_snapshot(
    const nanovec_mmap_header_state_t *state,
    nanovec_header_snapshot_t         *out_snap
);

nanovec_status_t nanovec_mmap_header_publish(
    nanovec_mmap_header_state_t     *state,
    nanovec_mmap_t                  *map,
    const nanovec_header_snapshot_t *next
);

nanovec_status_t nanovec_mmap_header_publish_count(
    nanovec_mmap_header_state_t *state,
    nanovec_mmap_t              *map,
    uint64_t                     vector_count,
    uint64_t                     max_capacity
);

#endif /* NANOVEC_STORAGE_MMAP_H */
