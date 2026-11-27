/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_WAL_H
#define NANOVEC_STORAGE_WAL_H

#include "core/types.h"
#include "storage/file.h"

#include <pthread.h>

#define NANOVEC_WAL_MAGIC_0       0x4EU /* 'N' */
#define NANOVEC_WAL_MAGIC_1       0x56U /* 'V' */
#define NANOVEC_WAL_MAGIC_2       0x57U /* 'W' */
#define NANOVEC_WAL_MAGIC_3       0x4CU /* 'L' */
#define NANOVEC_WAL_VERSION       1U
#define NANOVEC_WAL_HEADER_SIZE   32U
#define NANOVEC_WAL_RECORD_HDR_SZ 28U
#define NANOVEC_WAL_MAX_PAYLOAD   (64U * 1024U * 1024U)

typedef enum nanovec_wal_op {
    NANOVEC_WAL_OP_INSERT     = 1,
    NANOVEC_WAL_OP_UPDATE     = 2,
    NANOVEC_WAL_OP_DELETE     = 3,
    NANOVEC_WAL_OP_METADATA   = 4,
    NANOVEC_WAL_OP_CHECKPOINT = 5
} nanovec_wal_op_t;

typedef struct NANOVEC_PACKED nanovec_wal_file_header {
    uint8_t  magic[4];
    uint16_t version;
    uint16_t flags;
    uint64_t checkpoint_lsn;
    uint64_t reserved;
    uint32_t header_crc32;
    uint32_t pad;
} nanovec_wal_file_header_t;

NANOVEC_STATIC_ASSERT(sizeof(nanovec_wal_file_header_t) == NANOVEC_WAL_HEADER_SIZE, "wal hdr 32B");

typedef struct NANOVEC_PACKED nanovec_wal_record {
    uint64_t lsn;
    uint8_t  op_type;
    uint8_t  flags;
    uint16_t reserved;
    uint64_t vid;
    uint32_t len;
    uint32_t crc32;
    uint8_t  payload[];
} nanovec_wal_record_t;

NANOVEC_STATIC_ASSERT(sizeof(nanovec_wal_record_t) == NANOVEC_WAL_RECORD_HDR_SZ, "wal rec 28B");
NANOVEC_STATIC_ASSERT(offsetof(nanovec_wal_record_t, crc32) == 24U, "wal crc32 offset");

typedef struct nanovec_wal {
    nanovec_file_t  file;
    uint64_t        next_lsn;
    uint64_t        checkpoint_lsn;
    uint64_t        write_offset;
    bool            sync_on_write;
    bool            lock_init;
    pthread_mutex_t lock;
} nanovec_wal_t;

nanovec_status_t nanovec_wal_open(const char *path, bool sync_on_write, nanovec_wal_t *out_wal);
nanovec_status_t nanovec_wal_close(nanovec_wal_t *wal);

nanovec_status_t nanovec_wal_append(
    nanovec_wal_t   *wal,
    nanovec_wal_op_t op,
    nanovec_vid_t    vid,
    const void      *payload,
    uint32_t         len,
    uint64_t        *out_lsn
);

nanovec_status_t nanovec_wal_log_insert(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *vec_data,
    uint32_t       vec_bytes,
    uint64_t      *out_lsn
);

nanovec_status_t nanovec_wal_log_update(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *vec_data,
    uint32_t       vec_bytes,
    uint64_t      *out_lsn
);

nanovec_status_t nanovec_wal_log_delete(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    uint64_t      *out_lsn
);

nanovec_status_t nanovec_wal_log_metadata(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *meta,
    uint32_t       meta_len,
    uint64_t      *out_lsn
);

nanovec_status_t nanovec_wal_checkpoint(nanovec_wal_t *wal, uint64_t checkpoint_lsn);
nanovec_status_t nanovec_wal_truncate(nanovec_wal_t *wal, uint64_t checkpoint_lsn);
nanovec_status_t nanovec_wal_sync(nanovec_wal_t *wal);

#endif /* NANOVEC_STORAGE_WAL_H */
