/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/mmap.h"
#include "core/error.h"
#include "storage/checksum.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static size_t nanovec_os_page_size(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return (ps > 0 && nanovec_is_pow2((size_t)ps)) ? (size_t)ps : NANOVEC_PAGE_SIZE;
}

nanovec_status_t nanovec_mmap_open(
    const char     *path,
    size_t          min_size,
    bool            readonly,
    bool            create,
    nanovec_mmap_t *out_map
) {
    if (NANOVEC_UNLIKELY(path == NULL || path[0] == '\0' || out_map == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "mmap_open requires non-empty path and non-null out_map"
        );
    }
    if (NANOVEC_UNLIKELY(readonly && create)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "mmap_open cannot combine readonly and create"
        );
    }

    memset(out_map, 0, sizeof(*out_map));
    out_map->file.fd = -1;

    uint32_t fflags = 0U;
    if (readonly) fflags |= NANOVEC_FILE_READONLY;
    if (create)   fflags |= NANOVEC_FILE_CREATE;

    nanovec_status_t st = nanovec_file_open(path, fflags, &out_map->file);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    uint64_t fsize = 0ULL;
    st = nanovec_file_size(&out_map->file, &fsize);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_file_close(&out_map->file);
        return st;
    }

    if (!readonly && min_size > 0U && fsize < (uint64_t)min_size) {
        st = nanovec_file_allocate(&out_map->file, 0ULL, (uint64_t)min_size);
        if (st != NANOVEC_SUCCESS) {
            (void)nanovec_file_close(&out_map->file);
            return st;
        }
        fsize = (uint64_t)min_size;
    }

    if (NANOVEC_UNLIKELY(fsize == 0ULL)) {
        (void)nanovec_file_close(&out_map->file);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "cannot mmap empty file: %s",
            path
        );
    }

    if (NANOVEC_UNLIKELY(fsize > (uint64_t)SIZE_MAX)) {
        (void)nanovec_file_close(&out_map->file);
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_MMAP, "file exceeds address space");
    }

    int prot = readonly ? PROT_READ : (PROT_READ | PROT_WRITE);
    void *addr = mmap(NULL, (size_t)fsize, prot, MAP_SHARED, out_map->file.fd, 0);
    if (NANOVEC_UNLIKELY(addr == MAP_FAILED)) {
        int saved = errno;
        (void)nanovec_file_close(&out_map->file);
        return (nanovec_status_t)nanovec_set_errno(
            NANOVEC_ERR_MMAP,
            saved,
            "mmap(%s, %llu) failed",
            path,
            (unsigned long long)fsize
        );
    }

    out_map->base       = (uint8_t *)addr;
    out_map->mapped_len = (size_t)fsize;
    out_map->readonly   = readonly;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_mmap_open_nvec(
    const char                  *path,
    const nanovec_file_header_t *init_hdr,
    size_t                       initial_file_size,
    bool                         readonly,
    bool                         create,
    nanovec_mmap_t              *out_map
) {
    if (NANOVEC_UNLIKELY(path == NULL || out_map == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid mmap_open_nvec args");
    }

    size_t target_min = initial_file_size;
    if (create && target_min < NANOVEC_HEADER_SIZE) {
        target_min = NANOVEC_HEADER_SIZE;
    }

    bool existed_nonempty = false;
    {
        nanovec_file_t probe;
        if (nanovec_file_open(path, NANOVEC_FILE_READONLY, &probe) == NANOVEC_SUCCESS) {
            uint64_t sz = 0ULL;
            if (nanovec_file_size(&probe, &sz) == NANOVEC_SUCCESS && sz > 0ULL) {
                existed_nonempty = true;
            }
            (void)nanovec_file_close(&probe);
        }
    }

    nanovec_status_t st = nanovec_mmap_open(path, target_min, readonly, create, out_map);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    if (NANOVEC_UNLIKELY(out_map->mapped_len < NANOVEC_HEADER_SIZE)) {
        (void)nanovec_mmap_close(out_map);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "nvec file shorter than 128-byte header: %s",
            path
        );
    }

    nanovec_file_header_t *hdr = nanovec_mmap_header(out_map);
    if (!existed_nonempty && create && !readonly && init_hdr != NULL) {
        nanovec_file_header_t copy = *init_hdr;
        nanovec_header_seal(&copy);
        memcpy(hdr, &copy, sizeof(copy));
        st = nanovec_header_verify(hdr);
        if (st != NANOVEC_SUCCESS) {
            (void)nanovec_mmap_close(out_map);
            return st;
        }
        return NANOVEC_SUCCESS;
    }

    st = nanovec_header_verify(hdr);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_mmap_close(out_map);
        return st;
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_mmap_sync(const nanovec_mmap_t *map, bool async) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid mmap_sync handle");
    }
    if (map->readonly || map->mapped_len == 0U) {
        return NANOVEC_SUCCESS;
    }
    int flags = async ? MS_ASYNC : MS_SYNC;
    if (NANOVEC_UNLIKELY(msync(map->base, map->mapped_len, flags) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "msync() failed");
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_mmap_sync_range(
    const nanovec_mmap_t *map,
    uint64_t              offset,
    size_t                len,
    bool                  async
) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid mmap_sync_range handle");
    }
    if (map->readonly || len == 0U) {
        return NANOVEC_SUCCESS;
    }
    if (NANOVEC_UNLIKELY(offset > (uint64_t)map->mapped_len || len > map->mapped_len - (size_t)offset)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "mmap_sync_range out of bounds"
        );
    }

    size_t ps          = nanovec_os_page_size();
    size_t aligned_off = (size_t)offset & ~(ps - 1U);
    size_t aligned_len = len + ((size_t)offset - aligned_off);

    int flags = async ? MS_ASYNC : MS_SYNC;
    if (NANOVEC_UNLIKELY(msync(map->base + aligned_off, aligned_len, flags) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "msync(range) failed");
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_mmap_close(nanovec_mmap_t *map) {
    if (map == NULL) {
        return NANOVEC_SUCCESS;
    }

    nanovec_status_t st = NANOVEC_SUCCESS;
    if (map->base != NULL && map->mapped_len > 0U) {
        if (NANOVEC_UNLIKELY(munmap(map->base, map->mapped_len) != 0)) {
            st = (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_MMAP, errno, "munmap() failed");
        }
    }

    nanovec_status_t cst = nanovec_file_close(&map->file);
    if (st == NANOVEC_SUCCESS) {
        st = cst;
    }

    memset(map, 0, sizeof(*map));
    map->file.fd = -1;
    return st;
}

nanovec_status_t nanovec_madvise_seq(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    nanovec_status_t st = nanovec_mmap_validate_range(map, offset, len, "madvise_seq");
    if (st != NANOVEC_SUCCESS || len == 0U) {
        return st;
    }

    size_t ps          = nanovec_os_page_size();
    size_t aligned_off = (size_t)offset & ~(ps - 1U);
    size_t aligned_len = len + ((size_t)offset - aligned_off);

    st = nanovec_madvise_apply(map, aligned_off, aligned_len, MADV_SEQUENTIAL, "madvise_seq");
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    return nanovec_madvise_apply(map, aligned_off, aligned_len, MADV_WILLNEED, "madvise_seq");
}

nanovec_status_t nanovec_madvise_rand(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    return nanovec_madvise_range(map, offset, len, MADV_RANDOM, "madvise_rand");
}

nanovec_status_t nanovec_madvise_normal(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    return nanovec_madvise_range(map, offset, len, MADV_NORMAL, "madvise_normal");
}

nanovec_status_t nanovec_madvise_willneed(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    return nanovec_madvise_range(map, offset, len, MADV_WILLNEED, "madvise_willneed");
}

nanovec_status_t nanovec_madvise_dontneed(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    nanovec_status_t st = nanovec_mmap_validate_range(map, offset, len, "madvise_dontneed");
    if (st != NANOVEC_SUCCESS || len == 0U) {
        return st;
    }

    size_t ps        = nanovec_os_page_size();
    size_t raw_start = (size_t)offset;
    size_t raw_end   = raw_start + len;
    if (raw_start > SIZE_MAX - (ps - 1U)) {
        return NANOVEC_SUCCESS;
    }

    /* Shrink inward to full pages so adjacent hot data sharing a boundary page is not evicted. */
    size_t start = (raw_start + (ps - 1U)) & ~(ps - 1U);
    size_t end   = (raw_end == map->mapped_len) ? raw_end : (raw_end & ~(ps - 1U));
    if (start >= end) {
        return NANOVEC_SUCCESS;
    }

    size_t evict_len = end - start;
    st = nanovec_madvise_apply(map, start, evict_len, MADV_DONTNEED, "madvise_dontneed");
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
    if (map->file.fd >= 0 && (uint64_t)start <= (uint64_t)INT64_MAX &&
        (uint64_t)evict_len <= (uint64_t)INT64_MAX - (uint64_t)start) {
        (void)posix_fadvise(map->file.fd, (off_t)start, (off_t)evict_len, POSIX_FADV_DONTNEED);
    }
#endif
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_mmap_resident_pages(
    const nanovec_mmap_t *map,
    uint64_t              offset,
    size_t                len,
    size_t               *out_resident,
    size_t               *out_total
) {
    nanovec_status_t st = nanovec_mmap_validate_range(map, offset, len, "mmap_resident_pages");
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    if (len == 0U) {
        if (out_resident != NULL) *out_resident = 0U;
        if (out_total != NULL)    *out_total    = 0U;
        return NANOVEC_SUCCESS;
    }

    size_t ps          = nanovec_os_page_size();
    size_t aligned_off = (size_t)offset & ~(ps - 1U);
    size_t span        = len + ((size_t)offset - aligned_off);
    size_t total_pages = (span + (ps - 1U)) / ps;
    size_t resident    = 0U;

    unsigned char vec[1024];
    size_t        done = 0U;
    while (done < total_pages) {
        size_t chunk_pages = total_pages - done;
        if (chunk_pages > sizeof(vec)) {
            chunk_pages = sizeof(vec);
        }
        size_t chunk_off   = aligned_off + done * ps;
        size_t chunk_bytes = chunk_pages * ps;
        if (chunk_bytes > map->mapped_len - chunk_off) {
            chunk_bytes = map->mapped_len - chunk_off;
        }
        if (NANOVEC_UNLIKELY(mincore(map->base + chunk_off, chunk_bytes, vec) != 0)) {
            return (nanovec_status_t)nanovec_set_errno(
                NANOVEC_ERR_MMAP,
                errno,
                "mincore() failed"
            );
        }
        for (size_t i = 0U; i < chunk_pages; ++i) {
            resident += (size_t)(vec[i] & 1U);
        }
        done += chunk_pages;
    }

    if (out_resident != NULL) *out_resident = resident;
    if (out_total != NULL)    *out_total    = total_pages;
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_madvise_apply(
    const nanovec_mmap_t *map,
    uint64_t              offset,
    size_t                len,
    int                   advice
) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL || map->mapped_len == 0U)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid madvise handle");
    }

    size_t start = 0U;
    size_t span  = map->mapped_len;
    if (offset != 0ULL || len != 0U) {
        if (NANOVEC_UNLIKELY(
                offset > (uint64_t)map->mapped_len || len > map->mapped_len - (size_t)offset
            )) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_INVALID_ARG,
                "madvise range out of bounds"
            );
        }
        if (len == 0U) {
            return NANOVEC_SUCCESS;
        }
        size_t ps = nanovec_os_page_size();
        start = (size_t)offset & ~(ps - 1U);
        span  = ((size_t)offset + len) - start;
    }

    if (NANOVEC_UNLIKELY(madvise(map->base + start, span, advice) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_MMAP, errno, "madvise(%d) failed", advice);
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_madvise_seq(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    nanovec_status_t st = nanovec_madvise_apply(map, offset, len, MADV_SEQUENTIAL);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    return nanovec_madvise_apply(map, offset, len, MADV_WILLNEED);
}

nanovec_status_t nanovec_madvise_rand(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    return nanovec_madvise_apply(map, offset, len, MADV_RANDOM);
}

nanovec_status_t nanovec_madvise_normal(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    return nanovec_madvise_apply(map, offset, len, MADV_NORMAL);
}

nanovec_status_t nanovec_madvise_dontneed(const nanovec_mmap_t *map, uint64_t offset, size_t len) {
    if (NANOVEC_UNLIKELY(map == NULL || map->base == NULL || map->mapped_len == 0U)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid madvise_dontneed handle"
        );
    }

    size_t start = 0U;
    size_t span  = map->mapped_len;
    if (offset != 0ULL || len != 0U) {
        if (NANOVEC_UNLIKELY(
                offset > (uint64_t)map->mapped_len || len > map->mapped_len - (size_t)offset
            )) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_INVALID_ARG,
                "madvise_dontneed range out of bounds"
            );
        }
        if (len == 0U) {
            return NANOVEC_SUCCESS;
        }
        /* Evict only full interior pages to avoid touching adjacent hot pages (e.g. header). */
        size_t ps  = nanovec_os_page_size();
        size_t s   = nanovec_align_up((size_t)offset, ps);
        size_t end = ((size_t)offset + len) & ~(ps - 1U);
        if (end <= s) {
            return NANOVEC_SUCCESS;
        }
        start = s;
        span  = end - s;
    }

    if (NANOVEC_UNLIKELY(madvise(map->base + start, span, MADV_DONTNEED) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(
            NANOVEC_ERR_MMAP,
            errno,
            "madvise(MADV_DONTNEED) failed"
        );
    }
#ifdef POSIX_FADV_DONTNEED
    if (map->file.fd >= 0) {
        (void)posix_fadvise(map->file.fd, (off_t)start, (off_t)span, POSIX_FADV_DONTNEED);
    }
#endif
    return NANOVEC_SUCCESS;
}
