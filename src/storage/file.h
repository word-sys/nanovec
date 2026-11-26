/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_FILE_H
#define NANOVEC_STORAGE_FILE_H

#include "core/types.h"

#define NANOVEC_FILE_READONLY (1U << 0)
#define NANOVEC_FILE_CREATE   (1U << 1)
#define NANOVEC_FILE_TRUNCATE (1U << 2)
#define NANOVEC_FILE_DIRECT   (1U << 3)
#define NANOVEC_FILE_SYNC     (1U << 4)
#define NANOVEC_FILE_APPEND   (1U << 5)

typedef struct nanovec_file {
    int      fd;
    uint32_t flags;
    bool     direct_io;
    size_t   block_align;
} nanovec_file_t;

nanovec_status_t nanovec_file_open(const char *path, uint32_t flags, nanovec_file_t *out_file);
nanovec_status_t nanovec_file_close(nanovec_file_t *file);

nanovec_status_t nanovec_file_read_exact(const nanovec_file_t *file, void *buf, size_t len);
nanovec_status_t nanovec_file_write_exact(const nanovec_file_t *file, const void *buf, size_t len);
nanovec_status_t nanovec_file_pread_exact(
    const nanovec_file_t *file,
    uint64_t              offset,
    void                 *buf,
    size_t                len
);
nanovec_status_t nanovec_file_pwrite_exact(
    const nanovec_file_t *file,
    uint64_t              offset,
    const void           *buf,
    size_t                len
);

nanovec_status_t nanovec_file_seek(const nanovec_file_t *file, uint64_t offset);
nanovec_status_t nanovec_file_size(const nanovec_file_t *file, uint64_t *out_size);
nanovec_status_t nanovec_file_truncate(const nanovec_file_t *file, uint64_t new_size);
nanovec_status_t nanovec_file_allocate(const nanovec_file_t *file, uint64_t offset, uint64_t len);
nanovec_status_t nanovec_file_sync(const nanovec_file_t *file);
nanovec_status_t nanovec_file_datasync(const nanovec_file_t *file);

#endif /* NANOVEC_STORAGE_FILE_H */
