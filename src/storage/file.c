/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/file.h"
#include "core/error.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool nanovec_file_valid(const nanovec_file_t *file) {
    return file != NULL && file->fd >= 0;
}

nanovec_status_t nanovec_file_open(const char *path, uint32_t flags, nanovec_file_t *out_file) {
    if (NANOVEC_UNLIKELY(path == NULL || path[0] == '\0' || out_file == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "file_open requires non-empty path and non-null out_file"
        );
    }

    memset(out_file, 0, sizeof(*out_file));
    out_file->fd = -1;

    int oflags = O_CLOEXEC;
    if (flags & NANOVEC_FILE_READONLY) {
        if (NANOVEC_UNLIKELY(flags & (NANOVEC_FILE_CREATE | NANOVEC_FILE_TRUNCATE | NANOVEC_FILE_APPEND))) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_INVALID_ARG,
                "readonly file_open cannot combine create/truncate/append"
            );
        }
        oflags |= O_RDONLY;
    } else {
        oflags |= O_RDWR;
    }

    if (flags & NANOVEC_FILE_CREATE)   oflags |= O_CREAT;
    if (flags & NANOVEC_FILE_TRUNCATE) oflags |= O_TRUNC;
    if (flags & NANOVEC_FILE_SYNC)     oflags |= O_SYNC;
    if (flags & NANOVEC_FILE_APPEND)   oflags |= O_APPEND;

    bool want_direct = (flags & NANOVEC_FILE_DIRECT) != 0U;
    int  open_flags  = oflags;
#ifdef O_DIRECT
    if (want_direct) {
        open_flags |= O_DIRECT;
    }
#endif

    int fd = -1;
    do {
        fd = open(path, open_flags, 0644);
    } while (fd < 0 && errno == EINTR);

    bool direct_granted = want_direct;
#ifdef O_DIRECT
    /* Some filesystems (e.g., tmpfs/overlayfs) reject O_DIRECT with EINVAL. */
    if (fd < 0 && want_direct && errno == EINVAL) {
        direct_granted = false;
        do {
            fd = open(path, oflags, 0644);
        } while (fd < 0 && errno == EINTR);
    }
#else
    direct_granted = false;
#endif

    if (NANOVEC_UNLIKELY(fd < 0)) {
        int code = (errno == ENOENT) ? NANOVEC_ERR_NOT_FOUND : NANOVEC_ERR_IO;
        return (nanovec_status_t)nanovec_set_errno(code, errno, "open(%s) failed", path);
    }

    size_t align = NANOVEC_PAGE_SIZE;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_blksize > 0 && nanovec_is_pow2((size_t)st.st_blksize)) {
        align = (size_t)st.st_blksize;
    }

    out_file->fd          = fd;
    out_file->flags       = flags;
    out_file->direct_io   = direct_granted;
    out_file->block_align = align;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_close(nanovec_file_t *file) {
    if (file == NULL || file->fd < 0) {
        return NANOVEC_SUCCESS;
    }

    int fd = file->fd;
    file->fd        = -1;
    file->flags     = 0U;
    file->direct_io = false;

    if (NANOVEC_UNLIKELY(close(fd) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "close(fd=%d) failed", fd);
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_read_exact(const nanovec_file_t *file, void *buf, size_t len) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file) || (len > 0U && buf == NULL))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_read_exact args");
    }

    uint8_t *dst = (uint8_t *)buf;
    size_t   done = 0U;
    while (done < len) {
        ssize_t r = read(file->fd, dst + done, len - done);
        if (r > 0) {
            done += (size_t)r;
        } else if (r == 0) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_IO,
                "unexpected EOF after %zu of %zu bytes",
                done,
                len
            );
        } else if (errno != EINTR) {
            return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "read() failed");
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_write_exact(const nanovec_file_t *file, const void *buf, size_t len) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file) || (len > 0U && buf == NULL))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_write_exact args");
    }

    const uint8_t *src = (const uint8_t *)buf;
    size_t         done = 0U;
    while (done < len) {
        ssize_t w = write(file->fd, src + done, len - done);
        if (w > 0) {
            done += (size_t)w;
        } else if (w == 0) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_IO,
                "short write stalled after %zu of %zu bytes",
                done,
                len
            );
        } else if (errno != EINTR) {
            return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "write() failed");
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_pread_exact(
    const nanovec_file_t *file,
    uint64_t              offset,
    void                 *buf,
    size_t                len
) {
    if (NANOVEC_UNLIKELY(
            !nanovec_file_valid(file) || (len > 0U && buf == NULL) ||
            offset > (uint64_t)INT64_MAX || (uint64_t)len > (uint64_t)INT64_MAX - offset
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_pread_exact args");
    }

    uint8_t *dst  = (uint8_t *)buf;
    size_t   done = 0U;
    while (done < len) {
        ssize_t r = pread(file->fd, dst + done, len - done, (off_t)(offset + done));
        if (r > 0) {
            done += (size_t)r;
        } else if (r == 0) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_IO,
                "unexpected EOF in pread at offset %llu (%zu/%zu bytes)",
                (unsigned long long)(offset + done),
                done,
                len
            );
        } else if (errno != EINTR) {
            return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "pread() failed");
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_pwrite_exact(
    const nanovec_file_t *file,
    uint64_t              offset,
    const void           *buf,
    size_t                len
) {
    if (NANOVEC_UNLIKELY(
            !nanovec_file_valid(file) || (len > 0U && buf == NULL) ||
            offset > (uint64_t)INT64_MAX || (uint64_t)len > (uint64_t)INT64_MAX - offset
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_pwrite_exact args");
    }

    const uint8_t *src  = (const uint8_t *)buf;
    size_t         done = 0U;
    while (done < len) {
        ssize_t w = pwrite(file->fd, src + done, len - done, (off_t)(offset + done));
        if (w > 0) {
            done += (size_t)w;
        } else if (w == 0) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_IO,
                "pwrite stalled at offset %llu (%zu/%zu bytes)",
                (unsigned long long)(offset + done),
                done,
                len
            );
        } else if (errno != EINTR) {
            return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "pwrite() failed");
        }
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_seek(const nanovec_file_t *file, uint64_t offset) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file) || offset > (uint64_t)INT64_MAX)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_seek args");
    }
    if (NANOVEC_UNLIKELY(lseek(file->fd, (off_t)offset, SEEK_SET) == (off_t)-1)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "lseek() failed");
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_size(const nanovec_file_t *file, uint64_t *out_size) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file) || out_size == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_size args");
    }
    struct stat st;
    if (NANOVEC_UNLIKELY(fstat(file->fd, &st) != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "fstat() failed");
    }
    *out_size = st.st_size > 0 ? (uint64_t)st.st_size : 0ULL;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_truncate(const nanovec_file_t *file, uint64_t new_size) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file) || new_size > (uint64_t)INT64_MAX)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_truncate args");
    }
    int rc;
    do {
        rc = ftruncate(file->fd, (off_t)new_size);
    } while (rc != 0 && errno == EINTR);

    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "ftruncate() failed");
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_allocate(const nanovec_file_t *file, uint64_t offset, uint64_t len) {
    if (NANOVEC_UNLIKELY(
            !nanovec_file_valid(file) || offset > (uint64_t)INT64_MAX ||
            len > (uint64_t)INT64_MAX - offset
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_allocate args");
    }
    if (len == 0U) {
        return NANOVEC_SUCCESS;
    }

    int rc = posix_fallocate(file->fd, (off_t)offset, (off_t)len);
    if (rc == 0) {
        return NANOVEC_SUCCESS;
    }
    if (rc == EOPNOTSUPP || rc == EINVAL) {
        uint64_t cur = 0;
        nanovec_status_t st = nanovec_file_size(file, &cur);
        if (st != NANOVEC_SUCCESS) {
            return st;
        }
        uint64_t target = offset + len;
        if (target > cur) {
            return nanovec_file_truncate(file, target);
        }
        return NANOVEC_SUCCESS;
    }
    return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, rc, "posix_fallocate() failed");
}

nanovec_status_t nanovec_file_sync(const nanovec_file_t *file) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_sync handle");
    }
    int rc;
    do {
        rc = fsync(file->fd);
    } while (rc != 0 && errno == EINTR);

    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "fsync() failed");
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_file_datasync(const nanovec_file_t *file) {
    if (NANOVEC_UNLIKELY(!nanovec_file_valid(file))) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid file_datasync handle");
    }
    int rc;
    do {
        rc = fdatasync(file->fd);
    } while (rc != 0 && errno == EINTR);

    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_IO, errno, "fdatasync() failed");
    }
    return NANOVEC_SUCCESS;
}
