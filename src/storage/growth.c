/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/growth.h"
#include "core/error.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define NANOVEC_GROWTH_DOUBLE_THRESHOLD (256U * 1024U * 1024U)

static size_t nanovec_growth_page_size(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return (ps > 0 && nanovec_is_pow2((size_t)ps)) ? (size_t)ps : NANOVEC_PAGE_SIZE;
}

static size_t nanovec_growth_align_page(size_t n, size_t ps) {
    if (n == 0U) {
        return ps;
    }
    if (n > SIZE_MAX - (ps - 1U)) {
        return 0U;
    }
    return (n + (ps - 1U)) & ~(ps - 1U);
}

size_t nanovec_growth_calc_size(size_t current_size, size_t required_size, size_t page_size) {
    size_t ps = (page_size > 0U && nanovec_is_pow2(page_size)) ? page_size : NANOVEC_PAGE_SIZE;
    if (required_size <= current_size) {
        return current_size;
    }

    size_t target = (current_size > 0U) ? current_size : ps;
    while (target < required_size) {
        size_t next;
        if (target < NANOVEC_GROWTH_DOUBLE_THRESHOLD) {
            if (target > SIZE_MAX / 2U) {
                target = required_size;
                break;
            }
            next = target * 2U;
        } else {
            size_t half = target / 2U;
            if (target > SIZE_MAX - half) {
                target = required_size;
                break;
            }
            next = target + half;
        }
        if (next <= target) {
            target = required_size;
            break;
        }
        target = next;
    }

    return nanovec_growth_align_page(target, ps);
}

nanovec_status_t nanovec_mmap_grow(nanovec_mmap_t *map, size_t new_size, bool *out_moved) {
    if (out_moved != NULL) {
        *out_moved = false;
    }
    if (NANOVEC_UNLIKELY(
            map == NULL || map->base == NULL || map->mapped_len == 0U || map->file.fd < 0
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid mmap_grow handle");
    }
    if (NANOVEC_UNLIKELY(map->readonly)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "cannot grow read-only memory mapping"
        );
    }
    if (new_size <= map->mapped_len) {
        return NANOVEC_SUCCESS;
    }

    size_t ps      = nanovec_growth_page_size();
    size_t aligned = nanovec_growth_align_page(new_size, ps);
    if (NANOVEC_UNLIKELY(aligned == 0U || (uint64_t)aligned > (uint64_t)INT64_MAX)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "mmap_grow size overflow");
    }

    size_t old_len  = map->mapped_len;
    void  *old_base = map->base;

    nanovec_status_t st = nanovec_file_truncate(&map->file, (uint64_t)aligned);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

#if defined(__linux__)
    void *new_base = mremap(old_base, old_len, aligned, MREMAP_MAYMOVE);
    if (NANOVEC_UNLIKELY(new_base == MAP_FAILED)) {
        int saved = errno;
        (void)nanovec_file_truncate(&map->file, (uint64_t)old_len);
        return (nanovec_status_t)nanovec_set_errno(
            NANOVEC_ERR_MMAP,
            saved,
            "mremap(%zu -> %zu) failed",
            old_len,
            aligned
        );
    }
#else
    void *new_base = mmap(NULL, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, map->file.fd, 0);
    if (NANOVEC_UNLIKELY(new_base == MAP_FAILED)) {
        int saved = errno;
        (void)nanovec_file_truncate(&map->file, (uint64_t)old_len);
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_MMAP, saved, "mmap remap failed");
    }
    (void)munmap(old_base, old_len);
#endif

    map->base       = (uint8_t *)new_base;
    map->mapped_len = aligned;
    if (out_moved != NULL) {
        *out_moved = (new_base != old_base);
    }
    return NANOVEC_SUCCESS;
}

static void nanovec_growth_refresh_ptrs_locked(nanovec_growable_map_t *gmap, uint64_t gen) {
    nanovec_growth_ptrs_t p;
    memset(&p, 0, sizeof(p));
    p.base       = gmap->map.base;
    p.mapped_len = gmap->map.mapped_len;
    p.generation = gen;

    if (p.base != NULL && p.mapped_len >= NANOVEC_HEADER_SIZE &&
        p.base[0] == NANOVEC_FILE_MAGIC_0 && p.base[1] == NANOVEC_FILE_MAGIC_1 &&
        p.base[2] == NANOVEC_FILE_MAGIC_2 && p.base[3] == NANOVEC_FILE_MAGIC_3) {
        p.header   = nanovec_mmap_header(&gmap->map);
        p.vectors  = nanovec_mmap_vectors_ptr(&gmap->map);
        p.quant    = nanovec_mmap_quant_ptr(&gmap->map);
        p.graph    = nanovec_mmap_graph_ptr(&gmap->map);
        p.metadata = nanovec_mmap_metadata_ptr(&gmap->map);
    }

    gmap->ptrs = p;
}

nanovec_status_t nanovec_growth_init(nanovec_growable_map_t *gmap, nanovec_mmap_t map) {
    if (NANOVEC_UNLIKELY(gmap == NULL || map.base == NULL || map.mapped_len == 0U)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid growth_init args"
        );
    }

    memset(gmap, 0, sizeof(*gmap));
    gmap->map = map;

    pthread_rwlockattr_t attr;
    bool attr_ok = (pthread_rwlockattr_init(&attr) == 0);
#if defined(__GLIBC__) && defined(PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
    if (attr_ok) {
        (void)pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    }
#endif
    int rc = pthread_rwlock_init(&gmap->rwlock, attr_ok ? &attr : NULL);
    if (attr_ok) {
        (void)pthread_rwlockattr_destroy(&attr);
    }
    if (NANOVEC_UNLIKELY(rc != 0)) {
        gmap->map.base       = NULL;
        gmap->map.mapped_len = 0U;
        gmap->map.file.fd    = -1;
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INTERNAL, "rwlock init failed");
    }

    gmap->lock_init = true;
    atomic_init(&gmap->generation, 1ULL);
    nanovec_growth_refresh_ptrs_locked(gmap, 1ULL);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_growth_destroy(nanovec_growable_map_t *gmap) {
    if (gmap == NULL) {
        return NANOVEC_SUCCESS;
    }

    nanovec_status_t st = nanovec_mmap_close(&gmap->map);
    if (gmap->lock_init) {
        (void)pthread_rwlock_destroy(&gmap->rwlock);
        gmap->lock_init = false;
    }
    memset(&gmap->ptrs, 0, sizeof(gmap->ptrs));
    atomic_store_explicit(&gmap->generation, 0ULL, memory_order_relaxed);
    return st;
}

nanovec_status_t nanovec_growth_read_lock(
    nanovec_growable_map_t *gmap,
    nanovec_growth_ptrs_t  *out_ptrs
) {
    if (NANOVEC_UNLIKELY(gmap == NULL || !gmap->lock_init || out_ptrs == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid growth_read_lock args"
        );
    }

    int rc = pthread_rwlock_rdlock(&gmap->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock rdlock failed");
    }
    *out_ptrs = gmap->ptrs;
    return NANOVEC_SUCCESS;
}

void nanovec_growth_read_unlock(nanovec_growable_map_t *gmap) {
    if (NANOVEC_LIKELY(gmap != NULL && gmap->lock_init)) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
    }
}

nanovec_status_t nanovec_growth_ensure_capacity(
    nanovec_growable_map_t *gmap,
    size_t                  required_bytes,
    bool                   *out_grew
) {
    if (out_grew != NULL) {
        *out_grew = false;
    }
    if (NANOVEC_UNLIKELY(gmap == NULL || !gmap->lock_init)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid growth_ensure_capacity handle"
        );
    }

    /* Fast path under shared read lock when capacity is already sufficient. */
    if (pthread_rwlock_rdlock(&gmap->rwlock) == 0) {
        bool sufficient = (required_bytes <= gmap->map.mapped_len);
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        if (sufficient) {
            return NANOVEC_SUCCESS;
        }
    }

    int rc = pthread_rwlock_wrlock(&gmap->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock wrlock failed");
    }

    if (required_bytes <= gmap->map.mapped_len) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        return NANOVEC_SUCCESS;
    }

    size_t ps       = nanovec_growth_page_size();
    size_t new_size = nanovec_growth_calc_size(gmap->map.mapped_len, required_bytes, ps);
    if (NANOVEC_UNLIKELY(new_size == 0U)) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "growth target overflow");
    }

    nanovec_status_t st = nanovec_mmap_grow(&gmap->map, new_size, NULL);
    if (st == NANOVEC_SUCCESS) {
        uint64_t next_gen = atomic_fetch_add_explicit(&gmap->generation, 1ULL, memory_order_acq_rel) + 1ULL;
        nanovec_growth_refresh_ptrs_locked(gmap, next_gen);
        if (out_grew != NULL) {
            *out_grew = true;
        }
    }

    (void)pthread_rwlock_unlock(&gmap->rwlock);
    return st;
}

nanovec_status_t nanovec_growth_grow_exact(
    nanovec_growable_map_t *gmap,
    size_t                  new_size,
    bool                   *out_moved
) {
    if (out_moved != NULL) {
        *out_moved = false;
    }
    if (NANOVEC_UNLIKELY(gmap == NULL || !gmap->lock_init)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid growth_grow_exact handle"
        );
    }

    int rc = pthread_rwlock_wrlock(&gmap->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock wrlock failed");
    }

    size_t           old_len = gmap->map.mapped_len;
    nanovec_status_t st      = nanovec_mmap_grow(&gmap->map, new_size, out_moved);
    if (st == NANOVEC_SUCCESS && gmap->map.mapped_len != old_len) {
        uint64_t next_gen = atomic_fetch_add_explicit(&gmap->generation, 1ULL, memory_order_acq_rel) + 1ULL;
        nanovec_growth_refresh_ptrs_locked(gmap, next_gen);
    }

    (void)pthread_rwlock_unlock(&gmap->rwlock);
    return st;
}

nanovec_status_t nanovec_growth_refresh_ptrs(nanovec_growable_map_t *gmap) {
    if (NANOVEC_UNLIKELY(gmap == NULL || !gmap->lock_init)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid growth_refresh_ptrs handle"
        );
    }

    int rc = pthread_rwlock_wrlock(&gmap->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock wrlock failed");
    }

    uint64_t next_gen = atomic_fetch_add_explicit(&gmap->generation, 1ULL, memory_order_acq_rel) + 1ULL;
    nanovec_growth_refresh_ptrs_locked(gmap, next_gen);

    (void)pthread_rwlock_unlock(&gmap->rwlock);
    return NANOVEC_SUCCESS;
}
