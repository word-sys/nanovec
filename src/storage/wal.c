/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/wal.h"
#include "core/alloc.h"
#include "core/error.h"
#include "storage/checksum.h"

#include <string.h>

#define NANOVEC_WAL_STACK_BUF_SZ 8192U

static void nanovec_wal_hdr_encode(nanovec_wal_file_header_t *hdr, uint64_t checkpoint_lsn) {
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic[0]       = NANOVEC_WAL_MAGIC_0;
    hdr->magic[1]       = NANOVEC_WAL_MAGIC_1;
    hdr->magic[2]       = NANOVEC_WAL_MAGIC_2;
    hdr->magic[3]       = NANOVEC_WAL_MAGIC_3;
    hdr->version        = nanovec_htole16((uint16_t)NANOVEC_WAL_VERSION);
    hdr->flags          = nanovec_htole16(0x0001U);
    hdr->checkpoint_lsn = nanovec_htole64(checkpoint_lsn);
    uint32_t crc        = nanovec_crc32c(hdr, offsetof(nanovec_wal_file_header_t, header_crc32));
    hdr->header_crc32   = nanovec_htole32(crc);
}

static nanovec_status_t nanovec_wal_hdr_verify(
    const nanovec_wal_file_header_t *hdr,
    uint64_t                        *out_ckpt_lsn
) {
    if (NANOVEC_UNLIKELY(
            hdr->magic[0] != NANOVEC_WAL_MAGIC_0 || hdr->magic[1] != NANOVEC_WAL_MAGIC_1 ||
            hdr->magic[2] != NANOVEC_WAL_MAGIC_2 || hdr->magic[3] != NANOVEC_WAL_MAGIC_3
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_WAL, "invalid WAL magic");
    }
    uint16_t ver = nanovec_le16toh(hdr->version);
    if (NANOVEC_UNLIKELY(ver != NANOVEC_WAL_VERSION)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_VERSION,
            "unsupported WAL version %u",
            (unsigned)ver
        );
    }
    uint32_t stored   = nanovec_le32toh(hdr->header_crc32);
    uint32_t computed = nanovec_crc32c(hdr, offsetof(nanovec_wal_file_header_t, header_crc32));
    if (NANOVEC_UNLIKELY(stored != computed)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_WAL, "WAL header CRC-32C mismatch");
    }
    if (out_ckpt_lsn != NULL) {
        *out_ckpt_lsn = nanovec_le64toh(hdr->checkpoint_lsn);
    }
    return NANOVEC_SUCCESS;
}

static bool nanovec_wal_op_valid(uint8_t op) {
    return op >= (uint8_t)NANOVEC_WAL_OP_INSERT && op <= (uint8_t)NANOVEC_WAL_OP_CHECKPOINT;
}

static uint32_t nanovec_wal_record_crc(
    const uint8_t *hdr_bytes,
    const void    *payload,
    uint32_t       len
) {
    uint32_t crc = nanovec_crc32c_update(0U, hdr_bytes, 24U);
    if (len > 0U && payload != NULL) {
        crc = nanovec_crc32c_update(crc, payload, len);
    }
    return crc;
}

static nanovec_status_t nanovec_wal_scan_tail(nanovec_wal_t *wal, uint64_t fsize) {
    uint64_t pos      = NANOVEC_WAL_HEADER_SIZE;
    uint64_t prev_lsn = 0ULL;
    uint64_t max_lsn  = wal->checkpoint_lsn;

    uint8_t *scratch     = NULL;
    size_t   scratch_cap = 0U;

    while (fsize >= pos && fsize - pos >= NANOVEC_WAL_RECORD_HDR_SZ) {
        uint8_t raw_hdr[NANOVEC_WAL_RECORD_HDR_SZ];
        if (nanovec_file_pread_exact(&wal->file, pos, raw_hdr, sizeof(raw_hdr)) != NANOVEC_SUCCESS) {
            break;
        }

        uint64_t lsn    = nanovec_read_le64(raw_hdr + 0x00);
        uint8_t  op     = raw_hdr[0x08];
        uint32_t len    = nanovec_read_le32(raw_hdr + 0x14);
        uint32_t stored = nanovec_read_le32(raw_hdr + 0x18);

        if (lsn == 0ULL || (prev_lsn > 0ULL && lsn <= prev_lsn) ||
            !nanovec_wal_op_valid(op) || len > NANOVEC_WAL_MAX_PAYLOAD) {
            break;
        }
        uint64_t rec_total = (uint64_t)NANOVEC_WAL_RECORD_HDR_SZ + (uint64_t)len;
        if (rec_total > fsize - pos) {
            break;
        }

        if (len > scratch_cap) {
            nanovec_aligned_free(scratch);
            scratch_cap = nanovec_align_up((size_t)len, NANOVEC_CACHE_LINE);
            scratch     = (uint8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, scratch_cap);
            if (scratch == NULL) {
                return NANOVEC_ERR_OOM;
            }
        }
        if (len > 0U) {
            if (nanovec_file_pread_exact(
                    &wal->file, pos + NANOVEC_WAL_RECORD_HDR_SZ, scratch, len
                ) != NANOVEC_SUCCESS) {
                break;
            }
        }

        uint32_t computed = nanovec_wal_record_crc(raw_hdr, scratch, len);
        if (computed != stored) {
            break;
        }

        prev_lsn = lsn;
        if (lsn > max_lsn) {
            max_lsn = lsn;
        }
        pos += rec_total;
    }

    nanovec_aligned_free(scratch);
    wal->write_offset = pos;
    wal->next_lsn     = max_lsn + 1ULL;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_wal_open(const char *path, bool sync_on_write, nanovec_wal_t *out_wal) {
    if (NANOVEC_UNLIKELY(path == NULL || path[0] == '\0' || out_wal == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "wal_open requires non-empty path and non-null out_wal"
        );
    }

    memset(out_wal, 0, sizeof(*out_wal));
    out_wal->file.fd       = -1;
    out_wal->sync_on_write = sync_on_write;

    if (pthread_mutex_init(&out_wal->lock, NULL) != 0) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INTERNAL, "wal mutex init failed");
    }
    out_wal->lock_init = true;

    nanovec_status_t st = nanovec_file_open(path, NANOVEC_FILE_CREATE, &out_wal->file);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_wal_close(out_wal);
        return st;
    }

    uint64_t fsize = 0ULL;
    st = nanovec_file_size(&out_wal->file, &fsize);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_wal_close(out_wal);
        return st;
    }

    if (fsize == 0ULL) {
        nanovec_wal_file_header_t hdr;
        nanovec_wal_hdr_encode(&hdr, 0ULL);
        st = nanovec_file_pwrite_exact(&out_wal->file, 0ULL, &hdr, sizeof(hdr));
        if (st == NANOVEC_SUCCESS && sync_on_write) {
            st = nanovec_file_sync(&out_wal->file);
        }
        if (st != NANOVEC_SUCCESS) {
            (void)nanovec_wal_close(out_wal);
            return st;
        }
        out_wal->checkpoint_lsn = 0ULL;
        out_wal->next_lsn       = 1ULL;
        out_wal->write_offset   = NANOVEC_WAL_HEADER_SIZE;
        return NANOVEC_SUCCESS;
    }

    if (NANOVEC_UNLIKELY(fsize < NANOVEC_WAL_HEADER_SIZE)) {
        (void)nanovec_wal_close(out_wal);
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_WAL, "WAL file truncated below header");
    }

    nanovec_wal_file_header_t hdr;
    st = nanovec_file_pread_exact(&out_wal->file, 0ULL, &hdr, sizeof(hdr));
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_wal_close(out_wal);
        return st;
    }

    uint64_t ckpt_lsn = 0ULL;
    st = nanovec_wal_hdr_verify(&hdr, &ckpt_lsn);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_wal_close(out_wal);
        return st;
    }

    out_wal->checkpoint_lsn = ckpt_lsn;
    st = nanovec_wal_scan_tail(out_wal, fsize);
    if (st != NANOVEC_SUCCESS) {
        (void)nanovec_wal_close(out_wal);
        return st;
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_wal_close(nanovec_wal_t *wal) {
    if (wal == NULL) {
        return NANOVEC_SUCCESS;
    }

    nanovec_status_t st = nanovec_file_close(&wal->file);
    if (wal->lock_init) {
        (void)pthread_mutex_destroy(&wal->lock);
        wal->lock_init = false;
    }
    memset(wal, 0, sizeof(*wal));
    wal->file.fd = -1;
    return st;
}

nanovec_status_t nanovec_wal_append(
    nanovec_wal_t   *wal,
    nanovec_wal_op_t op,
    nanovec_vid_t    vid,
    const void      *payload,
    uint32_t         len,
    uint64_t        *out_lsn
) {
    if (NANOVEC_UNLIKELY(
            wal == NULL || !wal->lock_init || wal->file.fd < 0 ||
            !nanovec_wal_op_valid((uint8_t)op) || (len > 0U && payload == NULL) ||
            len > NANOVEC_WAL_MAX_PAYLOAD
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid wal_append args");
    }

    (void)pthread_mutex_lock(&wal->lock);

    uint64_t lsn = wal->next_lsn;
    uint8_t  raw_hdr[NANOVEC_WAL_RECORD_HDR_SZ];
    memset(raw_hdr, 0, sizeof(raw_hdr));
    nanovec_write_le64(raw_hdr + 0x00, lsn);
    raw_hdr[0x08] = (uint8_t)op;
    raw_hdr[0x09] = 0U;
    nanovec_write_le16(raw_hdr + 0x0A, 0U);
    nanovec_write_le64(raw_hdr + 0x0C, vid);
    nanovec_write_le32(raw_hdr + 0x14, len);
    uint32_t crc = nanovec_wal_record_crc(raw_hdr, payload, len);
    nanovec_write_le32(raw_hdr + 0x18, crc);

    size_t total = (size_t)NANOVEC_WAL_RECORD_HDR_SZ + (size_t)len;
    nanovec_status_t st;

    if (total <= NANOVEC_WAL_STACK_BUF_SZ) {
        uint8_t frame[NANOVEC_WAL_STACK_BUF_SZ];
        memcpy(frame, raw_hdr, sizeof(raw_hdr));
        if (len > 0U) {
            memcpy(frame + NANOVEC_WAL_RECORD_HDR_SZ, payload, len);
        }
        st = nanovec_file_pwrite_exact(&wal->file, wal->write_offset, frame, total);
    } else {
        uint8_t *heap_frame = (uint8_t *)nanovec_aligned_alloc(
            NANOVEC_CACHE_LINE, nanovec_align_up(total, NANOVEC_CACHE_LINE)
        );
        if (NANOVEC_UNLIKELY(heap_frame == NULL)) {
            (void)pthread_mutex_unlock(&wal->lock);
            return NANOVEC_ERR_OOM;
        }
        memcpy(heap_frame, raw_hdr, sizeof(raw_hdr));
        if (len > 0U) {
            memcpy(heap_frame + NANOVEC_WAL_RECORD_HDR_SZ, payload, len);
        }
        st = nanovec_file_pwrite_exact(&wal->file, wal->write_offset, heap_frame, total);
        nanovec_aligned_free(heap_frame);
    }

    if (st == NANOVEC_SUCCESS && wal->sync_on_write) {
        st = nanovec_file_datasync(&wal->file);
    }

    if (st == NANOVEC_SUCCESS) {
        wal->write_offset += (uint64_t)total;
        wal->next_lsn      = lsn + 1ULL;
        if (out_lsn != NULL) {
            *out_lsn = lsn;
        }
    }

    (void)pthread_mutex_unlock(&wal->lock);
    return st;
}

nanovec_status_t nanovec_wal_log_insert(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *vec_data,
    uint32_t       vec_bytes,
    uint64_t      *out_lsn
) {
    return nanovec_wal_append(wal, NANOVEC_WAL_OP_INSERT, vid, vec_data, vec_bytes, out_lsn);
}

nanovec_status_t nanovec_wal_log_update(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *vec_data,
    uint32_t       vec_bytes,
    uint64_t      *out_lsn
) {
    return nanovec_wal_append(wal, NANOVEC_WAL_OP_UPDATE, vid, vec_data, vec_bytes, out_lsn);
}

nanovec_status_t nanovec_wal_log_delete(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    uint64_t      *out_lsn
) {
    return nanovec_wal_append(wal, NANOVEC_WAL_OP_DELETE, vid, NULL, 0U, out_lsn);
}

nanovec_status_t nanovec_wal_log_metadata(
    nanovec_wal_t *wal,
    nanovec_vid_t  vid,
    const void    *meta,
    uint32_t       meta_len,
    uint64_t      *out_lsn
) {
    return nanovec_wal_append(wal, NANOVEC_WAL_OP_METADATA, vid, meta, meta_len, out_lsn);
}

nanovec_status_t nanovec_wal_checkpoint(nanovec_wal_t *wal, uint64_t checkpoint_lsn) {
    if (NANOVEC_UNLIKELY(wal == NULL || !wal->lock_init || wal->file.fd < 0)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid wal_checkpoint handle");
    }

    (void)pthread_mutex_lock(&wal->lock);

    if (NANOVEC_UNLIKELY(checkpoint_lsn < wal->checkpoint_lsn)) {
        (void)pthread_mutex_unlock(&wal->lock);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "wal_checkpoint cannot regress below current checkpoint_lsn"
        );
    }

    nanovec_wal_file_header_t hdr;
    nanovec_wal_hdr_encode(&hdr, checkpoint_lsn);
    nanovec_status_t st = nanovec_file_pwrite_exact(&wal->file, 0ULL, &hdr, sizeof(hdr));
    if (st == NANOVEC_SUCCESS) {
        st = nanovec_file_datasync(&wal->file);
    }
    if (st == NANOVEC_SUCCESS) {
        wal->checkpoint_lsn = checkpoint_lsn;
        if (wal->next_lsn <= checkpoint_lsn) {
            wal->next_lsn = checkpoint_lsn + 1ULL;
        }
    }

    (void)pthread_mutex_unlock(&wal->lock);
    return st;
}

nanovec_status_t nanovec_wal_truncate(nanovec_wal_t *wal, uint64_t checkpoint_lsn) {
    if (NANOVEC_UNLIKELY(wal == NULL || !wal->lock_init || wal->file.fd < 0)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid wal_truncate handle");
    }

    (void)pthread_mutex_lock(&wal->lock);

    if (NANOVEC_UNLIKELY(checkpoint_lsn < wal->checkpoint_lsn)) {
        (void)pthread_mutex_unlock(&wal->lock);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "wal_truncate cannot regress below current checkpoint_lsn"
        );
    }

    /* Persist checkpoint_lsn before shrinking file so crashes stay ordered. */
    nanovec_wal_file_header_t hdr;
    nanovec_wal_hdr_encode(&hdr, checkpoint_lsn);
    nanovec_status_t st = nanovec_file_pwrite_exact(&wal->file, 0ULL, &hdr, sizeof(hdr));
    if (st == NANOVEC_SUCCESS) {
        st = nanovec_file_datasync(&wal->file);
    }
    if (st == NANOVEC_SUCCESS) {
        st = nanovec_file_truncate(&wal->file, NANOVEC_WAL_HEADER_SIZE);
    }
    if (st == NANOVEC_SUCCESS) {
        st = nanovec_file_sync(&wal->file);
    }
    if (st == NANOVEC_SUCCESS) {
        wal->checkpoint_lsn = checkpoint_lsn;
        wal->write_offset   = NANOVEC_WAL_HEADER_SIZE;
        if (wal->next_lsn <= checkpoint_lsn) {
            wal->next_lsn = checkpoint_lsn + 1ULL;
        }
    }

    (void)pthread_mutex_unlock(&wal->lock);
    return st;
}

nanovec_status_t nanovec_wal_sync(nanovec_wal_t *wal) {
    if (NANOVEC_UNLIKELY(wal == NULL || !wal->lock_init || wal->file.fd < 0)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid wal_sync handle");
    }
    (void)pthread_mutex_lock(&wal->lock);
    nanovec_status_t st = nanovec_file_datasync(&wal->file);
    (void)pthread_mutex_unlock(&wal->lock);
    return st;
}

nanovec_status_t nanovec_wal_recover(
    nanovec_wal_t                *wal,
    nanovec_wal_replay_fn_t       replay_fn,
    void                         *ctx,
    nanovec_wal_recovery_stats_t *out_stats
) {
    if (NANOVEC_UNLIKELY(wal == NULL || !wal->lock_init || wal->file.fd < 0)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid wal_recover handle");
    }

    (void)pthread_mutex_lock(&wal->lock);

    uint64_t fsize = 0ULL;
    nanovec_status_t st = nanovec_file_size(&wal->file, &fsize);
    if (st != NANOVEC_SUCCESS) {
        (void)pthread_mutex_unlock(&wal->lock);
        return st;
    }
    if (NANOVEC_UNLIKELY(fsize < NANOVEC_WAL_HEADER_SIZE)) {
        (void)pthread_mutex_unlock(&wal->lock);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_WAL,
            "WAL file shorter than 32-byte header during recovery"
        );
    }

    nanovec_wal_file_header_t hdr;
    st = nanovec_file_pread_exact(&wal->file, 0ULL, &hdr, sizeof(hdr));
    if (st != NANOVEC_SUCCESS) {
        (void)pthread_mutex_unlock(&wal->lock);
        return st;
    }

    uint64_t ckpt_lsn = 0ULL;
    st = nanovec_wal_hdr_verify(&hdr, &ckpt_lsn);
    if (st != NANOVEC_SUCCESS) {
        (void)pthread_mutex_unlock(&wal->lock);
        return st;
    }
    wal->checkpoint_lsn = ckpt_lsn;

    nanovec_wal_recovery_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    stats.last_valid_lsn   = ckpt_lsn;
    stats.valid_end_offset = NANOVEC_WAL_HEADER_SIZE;

    uint64_t pos         = NANOVEC_WAL_HEADER_SIZE;
    uint64_t prev_lsn    = 0ULL;
    uint64_t max_lsn     = ckpt_lsn;
    uint8_t *scratch     = NULL;
    size_t   scratch_cap = 0U;

    while (pos < fsize) {
        if (fsize - pos < NANOVEC_WAL_RECORD_HDR_SZ) {
            stats.had_torn_tail = true;
            break;
        }

        uint8_t raw_hdr[NANOVEC_WAL_RECORD_HDR_SZ];
        st = nanovec_file_pread_exact(&wal->file, pos, raw_hdr, sizeof(raw_hdr));
        if (st != NANOVEC_SUCCESS) {
            nanovec_aligned_free(scratch);
            (void)pthread_mutex_unlock(&wal->lock);
            return st;
        }

        uint64_t lsn    = nanovec_read_le64(raw_hdr + 0x00);
        uint8_t  op     = raw_hdr[0x08];
        uint64_t vid    = nanovec_read_le64(raw_hdr + 0x0C);
        uint32_t len    = nanovec_read_le32(raw_hdr + 0x14);
        uint32_t stored = nanovec_read_le32(raw_hdr + 0x18);

        if (lsn == 0ULL || (prev_lsn > 0ULL && lsn <= prev_lsn) ||
            !nanovec_wal_op_valid(op) || len > NANOVEC_WAL_MAX_PAYLOAD) {
            stats.had_torn_tail = true;
            break;
        }

        uint64_t rec_total = (uint64_t)NANOVEC_WAL_RECORD_HDR_SZ + (uint64_t)len;
        if (rec_total > fsize - pos) {
            stats.had_torn_tail = true;
            break;
        }

        if (len > scratch_cap) {
            nanovec_aligned_free(scratch);
            scratch_cap = nanovec_align_up((size_t)len, NANOVEC_CACHE_LINE);
            scratch     = (uint8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, scratch_cap);
            if (NANOVEC_UNLIKELY(scratch == NULL)) {
                (void)pthread_mutex_unlock(&wal->lock);
                return NANOVEC_ERR_OOM;
            }
        }

        if (len > 0U) {
            st = nanovec_file_pread_exact(&wal->file, pos + NANOVEC_WAL_RECORD_HDR_SZ, scratch, len);
            if (st != NANOVEC_SUCCESS) {
                nanovec_aligned_free(scratch);
                (void)pthread_mutex_unlock(&wal->lock);
                return st;
            }
        }

        uint32_t computed = nanovec_wal_record_crc(raw_hdr, scratch, len);
        if (computed != stored) {
            stats.had_torn_tail = true;
            break;
        }

        stats.records_scanned++;
        if (lsn <= ckpt_lsn) {
            stats.records_skipped++;
        } else if (replay_fn != NULL) {
            const void *payload_ptr = (len > 0U) ? (const void *)scratch : NULL;
            st = replay_fn((nanovec_wal_op_t)op, lsn, (nanovec_vid_t)vid, payload_ptr, len, ctx);
            if (st != NANOVEC_SUCCESS) {
                nanovec_aligned_free(scratch);
                (void)pthread_mutex_unlock(&wal->lock);
                return st;
            }
            stats.records_replayed++;
        }

        prev_lsn = lsn;
        if (lsn > max_lsn) {
            max_lsn = lsn;
        }
        pos += rec_total;
    }

    nanovec_aligned_free(scratch);

    stats.last_valid_lsn   = max_lsn;
    stats.valid_end_offset = pos;

    if (pos < fsize) {
        stats.had_torn_tail   = true;
        stats.truncated_bytes = fsize - pos;
        st = nanovec_file_truncate(&wal->file, pos);
        if (st == NANOVEC_SUCCESS) {
            st = nanovec_file_sync(&wal->file);
        }
        if (st != NANOVEC_SUCCESS) {
            (void)pthread_mutex_unlock(&wal->lock);
            return st;
        }
    }

    wal->write_offset = pos;
    wal->next_lsn     = max_lsn + 1ULL;

    if (out_stats != NULL) {
        *out_stats = stats;
    }

    (void)pthread_mutex_unlock(&wal->lock);
    return NANOVEC_SUCCESS;
}
