/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/checksum.h"
#include "storage/file.h"
#include "storage/format.h"
#include "storage/growth.h"
#include "storage/mmap.h"
#include "storage/wal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static void make_temp_path(char *buf, size_t cap, const char *tag) {
    int n = snprintf(buf, cap, "/tmp/nanovec_storage_%ld_%s.nvec", (long)getpid(), tag);
    CHECK(n > 0 && (size_t)n < cap);
}

static void test_header_and_checksum(void) {
    const char *vec = "123456789";
    CHECK(nanovec_crc32c(vec, 9U) == 0xE3069283U);
    CHECK(nanovec_crc32c_scalar(0U, vec, 9U) == 0xE3069283U);

    uint32_t stream_crc = nanovec_crc32c_update(0U, vec, 4U);
    stream_crc          = nanovec_crc32c_update(stream_crc, vec + 4, 5U);
    CHECK(stream_crc == 0xE3069283U);

    CHECK(nanovec_xxh64(NULL, 0U, 0ULL) == 0xEF46DB3751D8E999ULL);

    nanovec_file_header_t hdr;
    nanovec_header_init(
        &hdr,
        384U,
        NANOVEC_METRIC_COSINE,
        NANOVEC_INDEX_HNSW,
        NANOVEC_QUANT_NONE,
        1024ULL
    );
    hdr.vectors_offset = nanovec_htole64(128ULL);
    nanovec_header_seal(&hdr);
    CHECK(nanovec_header_verify(&hdr) == NANOVEC_SUCCESS);

    hdr.dim = nanovec_htole16(768U);
    CHECK(nanovec_header_verify(&hdr) == NANOVEC_ERR_CORRUPT);

    CHECK(nanovec_vector_stride_bytes(384U, NANOVEC_QUANT_NONE, 0U) == 1536U);
    CHECK(nanovec_vector_stride_bytes(384U, NANOVEC_QUANT_SQ8, 0U) == 384U);
    CHECK(nanovec_vector_stride_bytes(384U, NANOVEC_QUANT_BQ, 0U) == 48U);
    CHECK(nanovec_vector_stride_bytes(384U, NANOVEC_QUANT_PQ, 48U) == 48U);
}

static void test_file_io(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "fileio");
    (void)unlink(path);

    nanovec_file_t f;
    CHECK(nanovec_file_open(path, NANOVEC_FILE_READONLY, &f) == NANOVEC_ERR_NOT_FOUND);
    CHECK(nanovec_file_open(path, NANOVEC_FILE_CREATE | NANOVEC_FILE_TRUNCATE, &f) == NANOVEC_SUCCESS);

    const uint8_t payload[16] = {
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
        0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x01
    };
    CHECK(nanovec_file_write_exact(&f, payload, sizeof(payload)) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_datasync(&f) == NANOVEC_SUCCESS);

    uint64_t sz = 0ULL;
    CHECK(nanovec_file_size(&f, &sz) == NANOVEC_SUCCESS);
    CHECK(sz == sizeof(payload));

    uint8_t out[16];
    memset(out, 0, sizeof(out));
    CHECK(nanovec_file_pread_exact(&f, 0ULL, out, sizeof(out)) == NANOVEC_SUCCESS);
    CHECK(memcmp(payload, out, sizeof(payload)) == 0);

    CHECK(nanovec_file_allocate(&f, 0ULL, 8192ULL) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_size(&f, &sz) == NANOVEC_SUCCESS);
    CHECK(sz == 8192ULL);

    CHECK(nanovec_file_close(&f) == NANOVEC_SUCCESS);
    CHECK(unlink(path) == 0);
}

static void test_mmap_and_madvise(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "mmap");
    (void)unlink(path);

    long   ps_raw    = sysconf(_SC_PAGESIZE);
    size_t page_size = (ps_raw > 0) ? (size_t)ps_raw : NANOVEC_PAGE_SIZE;
    size_t num_pages = 32U;
    size_t file_size = num_pages * page_size;

    nanovec_file_header_t init_hdr;
    nanovec_header_init(
        &init_hdr,
        64U,
        NANOVEC_METRIC_L2,
        NANOVEC_INDEX_FLAT,
        NANOVEC_QUANT_NONE,
        256ULL
    );
    init_hdr.vectors_offset = nanovec_htole64((uint64_t)page_size);

    nanovec_mmap_t map;
    CHECK(nanovec_mmap_open_nvec(path, &init_hdr, file_size, false, true, &map) == NANOVEC_SUCCESS);
    CHECK(map.mapped_len == file_size);
    CHECK(nanovec_mmap_vectors_ptr(&map) == (void *)(map.base + page_size));
    CHECK(nanovec_mmap_ptr(&map, file_size, 1U) == NULL);

    uint8_t *vecs = (uint8_t *)nanovec_mmap_vectors_ptr(&map);
    size_t   vec_bytes = file_size - page_size;
    for (size_t i = 0U; i < vec_bytes; ++i) {
        vecs[i] = (uint8_t)((i * 131U + 17U) & 0xFFU);
    }

    CHECK(nanovec_madvise_seq(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_seq(&map, 128ULL, vec_bytes) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_rand(&map, (uint64_t)page_size, vec_bytes) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_willneed(&map, (uint64_t)page_size, 4U * page_size) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_normal(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);

    /* Zero-length advise is a no-op within bounds. */
    CHECK(nanovec_madvise_seq(&map, 0ULL, 0U) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_rand(&map, (uint64_t)file_size, 0U) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_dontneed(&map, 0ULL, 0U) == NANOVEC_SUCCESS);

    /* Unaligned sub-page dontneed must safely no-op without harming page 0. */
    CHECK(nanovec_madvise_dontneed(&map, 128ULL, 256U) == NANOVEC_SUCCESS);

    /* Unaligned multi-page dontneed evicts interior pages while preserving MAP_SHARED writes. */
    CHECK(nanovec_madvise_dontneed(&map, 128ULL, file_size - 128U) == NANOVEC_SUCCESS);
    CHECK(nanovec_header_verify(nanovec_mmap_header(&map)) == NANOVEC_SUCCESS);
    for (size_t i = 0U; i < vec_bytes; ++i) {
        CHECK(vecs[i] == (uint8_t)((i * 131U + 17U) & 0xFFU));
    }

    size_t resident = 0U;
    size_t total    = 0U;
    CHECK(nanovec_mmap_resident_pages(&map, 0ULL, map.mapped_len, &resident, &total) == NANOVEC_SUCCESS);
    CHECK(total == num_pages);
    CHECK(resident == num_pages);

    CHECK(nanovec_mmap_resident_pages(&map, (uint64_t)page_size + 64ULL, 2U * page_size, &resident, &total) == NANOVEC_SUCCESS);
    CHECK(total == 3U);
    CHECK(resident == 3U);

    CHECK(nanovec_mmap_sync(&map, false) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_dontneed(&map, (uint64_t)page_size, vec_bytes) == NANOVEC_SUCCESS);

    /* Verify resident_pages accepts partial output pointers and valid bounds. */
    CHECK(nanovec_mmap_resident_pages(&map, (uint64_t)page_size, vec_bytes, &resident, NULL) == NANOVEC_SUCCESS);
    CHECK(resident <= num_pages - 1U);
    CHECK(nanovec_mmap_resident_pages(&map, 0ULL, 0U, &resident, &total) == NANOVEC_SUCCESS);
    CHECK(resident == 0U && total == 0U);

    /* Re-read via sequential prefetch and verify byte-exact data after eviction. */
    CHECK(nanovec_madvise_seq(&map, (uint64_t)page_size, vec_bytes) == NANOVEC_SUCCESS);
    for (size_t i = 0U; i < vec_bytes; ++i) {
        CHECK(vecs[i] == (uint8_t)((i * 131U + 17U) & 0xFFU));
    }

    /* Defensive error checks. */
    CHECK(nanovec_madvise_seq(NULL, 0ULL, 64U) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_madvise_rand(&map, (uint64_t)file_size + 1ULL, 0U) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_madvise_dontneed(&map, (uint64_t)file_size - 8ULL, 16U) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_madvise_normal(&map, 64ULL, SIZE_MAX) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_resident_pages(&map, (uint64_t)file_size, 1U, &resident, &total) == NANOVEC_ERR_INVALID_ARG);

    CHECK(nanovec_mmap_close(&map) == NANOVEC_SUCCESS);

    /* Re-open read-only and verify persistence + madvise on read-only mapping. */
    CHECK(nanovec_mmap_open_nvec(path, NULL, 0U, true, false, &map) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_rand(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_dontneed(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    const uint8_t *ro_vecs = (const uint8_t *)nanovec_mmap_vectors_ptr(&map);
    CHECK(ro_vecs != NULL);
    for (size_t i = 0U; i < vec_bytes; ++i) {
        CHECK(ro_vecs[i] == (uint8_t)((i * 131U + 17U) & 0xFFU));
    }
    CHECK(nanovec_mmap_close(&map) == NANOVEC_SUCCESS);
    CHECK(unlink(path) == 0);
}

static void test_madvise_small_file(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "small");
    (void)unlink(path);

    nanovec_file_header_t init_hdr;
    nanovec_header_init(&init_hdr, 8U, NANOVEC_METRIC_DOT, NANOVEC_INDEX_FLAT, NANOVEC_QUANT_NONE, 4ULL);

    nanovec_mmap_t map;
    CHECK(nanovec_mmap_open_nvec(path, &init_hdr, 256U, false, true, &map) == NANOVEC_SUCCESS);
    CHECK(map.mapped_len == 256U);
    CHECK(nanovec_madvise_seq(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_rand(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    CHECK(nanovec_madvise_dontneed(&map, 0ULL, map.mapped_len) == NANOVEC_SUCCESS);
    CHECK(nanovec_header_verify(nanovec_mmap_header(&map)) == NANOVEC_SUCCESS);
    CHECK(nanovec_mmap_close(&map) == NANOVEC_SUCCESS);
    CHECK(unlink(path) == 0);
}

typedef struct wal_thread_ctx {
    nanovec_wal_t *wal;
    uint32_t       tid;
    uint32_t       count;
    uint64_t       lsns[64];
} wal_thread_ctx_t;

static void *wal_worker_fn(void *arg) {
    wal_thread_ctx_t *ctx = (wal_thread_ctx_t *)arg;
    float             vec[8];
    for (uint32_t i = 0U; i < ctx->count; ++i) {
        for (uint32_t d = 0U; d < 8U; ++d) {
            vec[d] = (float)(ctx->tid * 100U + i) + 0.1f * (float)d;
        }
        uint64_t lsn = 0ULL;
        nanovec_status_t st = nanovec_wal_log_insert(
            ctx->wal,
            ((uint64_t)ctx->tid << 32) | (uint64_t)i,
            vec,
            (uint32_t)sizeof(vec),
            &lsn
        );
        CHECK(st == NANOVEC_SUCCESS);
        ctx->lsns[i] = lsn;
    }
    return NULL;
}

static void test_wal_append_and_reopen(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "wal_basic");
    (void)unlink(path);

    nanovec_wal_t wal;
    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    CHECK(wal.checkpoint_lsn == 0ULL);
    CHECK(wal.next_lsn == 1ULL);
    CHECK(wal.write_offset == NANOVEC_WAL_HEADER_SIZE);

    /* Verify on-disk 32B header magic and CRC-32C. */
    uint8_t raw_hdr[NANOVEC_WAL_HEADER_SIZE];
    CHECK(nanovec_file_pread_exact(&wal.file, 0ULL, raw_hdr, sizeof(raw_hdr)) == NANOVEC_SUCCESS);
    CHECK(raw_hdr[0] == 'N' && raw_hdr[1] == 'V' && raw_hdr[2] == 'W' && raw_hdr[3] == 'L');
    CHECK(nanovec_read_le32(raw_hdr + 24) == nanovec_crc32c(raw_hdr, 24U));

    /* Log insert, update, delete, metadata, and >8KiB large payload. */
    float vec1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const char *meta = "{\"title\":\"nanovec\"}";

    uint64_t lsn1 = 0ULL, lsn2 = 0ULL, lsn3 = 0ULL, lsn4 = 0ULL, lsn5 = 0ULL;
    CHECK(nanovec_wal_log_insert(&wal, 100ULL, vec1, (uint32_t)sizeof(vec1), &lsn1) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_log_update(&wal, 100ULL, vec2, (uint32_t)sizeof(vec2), &lsn2) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_log_delete(&wal, 100ULL, &lsn3) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_log_metadata(&wal, 101ULL, meta, (uint32_t)strlen(meta), &lsn4) == NANOVEC_SUCCESS);

    uint8_t *big_payload = (uint8_t *)malloc(12000U);
    CHECK(big_payload != NULL);
    for (size_t i = 0U; i < 12000U; ++i) {
        big_payload[i] = (uint8_t)((i * 73U + 19U) & 0xFFU);
    }
    CHECK(nanovec_wal_log_insert(&wal, 102ULL, big_payload, 12000U, &lsn5) == NANOVEC_SUCCESS);
    free(big_payload);

    CHECK(lsn1 == 1ULL && lsn2 == 2ULL && lsn3 == 3ULL && lsn4 == 4ULL && lsn5 == 5ULL);
    uint64_t expected_off = NANOVEC_WAL_HEADER_SIZE
        + (NANOVEC_WAL_RECORD_HDR_SZ + (uint64_t)sizeof(vec1))
        + (NANOVEC_WAL_RECORD_HDR_SZ + (uint64_t)sizeof(vec2))
        + (NANOVEC_WAL_RECORD_HDR_SZ + 0ULL)
        + (NANOVEC_WAL_RECORD_HDR_SZ + (uint64_t)strlen(meta))
        + (NANOVEC_WAL_RECORD_HDR_SZ + 12000ULL);
    CHECK(wal.write_offset == expected_off);
    CHECK(wal.next_lsn == 6ULL);

    /* Inspect exact binary framing + CRC-32C of the first record. */
    uint8_t rec1[NANOVEC_WAL_RECORD_HDR_SZ + 16U];
    CHECK(nanovec_file_pread_exact(&wal.file, NANOVEC_WAL_HEADER_SIZE, rec1, sizeof(rec1)) == NANOVEC_SUCCESS);
    CHECK(nanovec_read_le64(rec1 + 0x00) == 1ULL);
    CHECK(rec1[0x08] == (uint8_t)NANOVEC_WAL_OP_INSERT);
    CHECK(nanovec_read_le64(rec1 + 0x0C) == 100ULL);
    CHECK(nanovec_read_le32(rec1 + 0x14) == 16U);
    uint32_t rec1_crc = nanovec_crc32c_update(0U, rec1, 24U);
    rec1_crc          = nanovec_crc32c_update(rec1_crc, rec1 + 28, 16U);
    CHECK(nanovec_read_le32(rec1 + 0x18) == rec1_crc);
    CHECK(memcmp(rec1 + 28, vec1, sizeof(vec1)) == 0);

    /* Checkpoint advances monotonically; regression is rejected. */
    CHECK(nanovec_wal_checkpoint(&wal, 3ULL) == NANOVEC_SUCCESS);
    CHECK(wal.checkpoint_lsn == 3ULL);
    CHECK(nanovec_wal_checkpoint(&wal, 2ULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_wal_sync(&wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* Append a simulated torn tail (corrupt bytes past last valid record), then reopen. */
    nanovec_file_t rawf;
    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    const uint8_t torn[19] = {0xFF, 0x01, 0x02, 0x03, 0xAA, 0xBB, 0xCC, 0xDD};
    CHECK(nanovec_file_pwrite_exact(&rawf, expected_off, torn, sizeof(torn)) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);

    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    CHECK(wal.checkpoint_lsn == 3ULL);
    CHECK(wal.next_lsn == 6ULL);
    CHECK(wal.write_offset == expected_off);

    /* Truncate resets file back to 32B header and preserves monotonic LSNs. */
    CHECK(nanovec_wal_truncate(&wal, 5ULL) == NANOVEC_SUCCESS);
    CHECK(wal.checkpoint_lsn == 5ULL);
    CHECK(wal.write_offset == NANOVEC_WAL_HEADER_SIZE);
    CHECK(wal.next_lsn == 6ULL);
    uint64_t fsz = 0ULL;
    CHECK(nanovec_file_size(&wal.file, &fsz) == NANOVEC_SUCCESS);
    CHECK(fsz == NANOVEC_WAL_HEADER_SIZE);

    uint64_t lsn6 = 0ULL;
    CHECK(nanovec_wal_log_delete(&wal, 999ULL, &lsn6) == NANOVEC_SUCCESS);
    CHECK(lsn6 == 6ULL);

    /* Defensive invalid-argument checks. */
    CHECK(nanovec_wal_append(&wal, (nanovec_wal_op_t)0, 1ULL, NULL, 0U, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_wal_append(&wal, NANOVEC_WAL_OP_INSERT, 1ULL, NULL, 4U, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_wal_append(NULL, NANOVEC_WAL_OP_DELETE, 1ULL, NULL, 0U, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* Reopen after truncate+append confirms persisted state. */
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_SUCCESS);
    CHECK(wal.checkpoint_lsn == 5ULL);
    CHECK(wal.next_lsn == 7ULL);
    CHECK(wal.write_offset == NANOVEC_WAL_HEADER_SIZE + NANOVEC_WAL_RECORD_HDR_SZ);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* Corrupt header magic and CRC must be rejected on open. */
    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    uint8_t bad_byte = 0x58U;
    CHECK(nanovec_file_pwrite_exact(&rawf, 16ULL, &bad_byte, 1U) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_ERR_WAL);

    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_pwrite_exact(&rawf, 0ULL, &bad_byte, 1U) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_ERR_WAL);

    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_truncate(&rawf, 12ULL) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_ERR_WAL);

    CHECK(unlink(path) == 0);
}

static void test_wal_concurrent_appends(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "wal_mt");
    (void)unlink(path);

    nanovec_wal_t wal;
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_SUCCESS);

    enum { NTHREADS = 4, PER_THREAD = 50, TOTAL = NTHREADS * PER_THREAD };
    pthread_t        threads[NTHREADS];
    wal_thread_ctx_t ctxs[NTHREADS];

    for (uint32_t t = 0U; t < NTHREADS; ++t) {
        ctxs[t].wal   = &wal;
        ctxs[t].tid   = t;
        ctxs[t].count = PER_THREAD;
        CHECK(pthread_create(&threads[t], NULL, wal_worker_fn, &ctxs[t]) == 0);
    }
    for (uint32_t t = 0U; t < NTHREADS; ++t) {
        CHECK(pthread_join(threads[t], NULL) == 0);
    }

    uint8_t seen[TOTAL + 1];
    memset(seen, 0, sizeof(seen));
    for (uint32_t t = 0U; t < NTHREADS; ++t) {
        for (uint32_t i = 0U; i < PER_THREAD; ++i) {
            uint64_t lsn = ctxs[t].lsns[i];
            CHECK(lsn >= 1ULL && lsn <= (uint64_t)TOTAL);
            CHECK(seen[lsn] == 0U);
            seen[lsn] = 1U;
        }
    }
    CHECK(wal.next_lsn == (uint64_t)TOTAL + 1ULL);
    CHECK(nanovec_wal_sync(&wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* Reopen and verify all 200 concurrent records verify intact via CRC-32C. */
    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_SUCCESS);
    CHECK(wal.next_lsn == (uint64_t)TOTAL + 1ULL);
    CHECK(wal.write_offset == NANOVEC_WAL_HEADER_SIZE + (uint64_t)TOTAL * (NANOVEC_WAL_RECORD_HDR_SZ + 32ULL));
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);
    CHECK(unlink(path) == 0);
}

typedef struct replay_state {
    uint64_t         last_lsn;
    uint32_t         call_count;
    uint32_t         fail_on_call;
    bool             present[16];
    float            vecs[16][4];
    char             meta[16][64];
    uint32_t         big_len;
    uint8_t          big_first;
    uint8_t          big_last;
} replay_state_t;

static nanovec_status_t test_replay_cb(
    nanovec_wal_op_t op,
    uint64_t         lsn,
    nanovec_vid_t    vid,
    const void      *payload,
    uint32_t         len,
    void            *ctx
) {
    replay_state_t *st = (replay_state_t *)ctx;
    st->call_count++;
    if (st->fail_on_call > 0U && st->call_count == st->fail_on_call) {
        return NANOVEC_ERR_OOM;
    }
    CHECK(lsn > st->last_lsn);
    st->last_lsn = lsn;

    if (vid == 15ULL && op == NANOVEC_WAL_OP_INSERT) {
        CHECK(payload != NULL && len == 10000U);
        const uint8_t *bytes = (const uint8_t *)payload;
        st->big_len   = len;
        st->big_first = bytes[0];
        st->big_last  = bytes[len - 1U];
        return NANOVEC_SUCCESS;
    }

    CHECK(vid < 16ULL);
    switch (op) {
        case NANOVEC_WAL_OP_INSERT:
        case NANOVEC_WAL_OP_UPDATE:
            CHECK(payload != NULL && len == 4U * sizeof(float));
            memcpy(st->vecs[vid], payload, len);
            st->present[vid] = true;
            break;
        case NANOVEC_WAL_OP_DELETE:
            CHECK(payload == NULL && len == 0U);
            st->present[vid] = false;
            break;
        case NANOVEC_WAL_OP_METADATA:
            CHECK(payload != NULL && len > 0U && len < sizeof(st->meta[vid]));
            memcpy(st->meta[vid], payload, len);
            st->meta[vid][len] = '\0';
            break;
        case NANOVEC_WAL_OP_CHECKPOINT:
            break;
        default:
            return NANOVEC_ERR_WAL;
    }
    return NANOVEC_SUCCESS;
}

static void test_wal_recover_and_crash_replay(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "wal_recover");
    (void)unlink(path);

    /* 1. Empty WAL recovery succeeds cleanly. */
    nanovec_wal_t wal;
    nanovec_wal_recovery_stats_t stats;
    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_recover(&wal, test_replay_cb, NULL, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 0ULL && stats.records_replayed == 0ULL);
    CHECK(stats.records_skipped == 0ULL && stats.last_valid_lsn == 0ULL);
    CHECK(stats.valid_end_offset == NANOVEC_WAL_HEADER_SIZE);
    CHECK(!stats.had_torn_tail && stats.truncated_bytes == 0ULL);

    /* 2. Write 6 records, checkpoint at LSN 2 so LSNs 1-2 are skipped and 3-6 replayed. */
    float v1[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float v2[4]  = {10.0f, 20.0f, 30.0f, 40.0f};
    float v2u[4] = {11.0f, 22.0f, 33.0f, 44.0f};
    const char *m3 = "{\"tag\":\"recovered\"}";

    uint64_t lsn = 0ULL;
    CHECK(nanovec_wal_log_insert(&wal, 1ULL, v1, sizeof(v1), &lsn) == NANOVEC_SUCCESS && lsn == 1ULL);
    CHECK(nanovec_wal_log_insert(&wal, 2ULL, v2, sizeof(v2), &lsn) == NANOVEC_SUCCESS && lsn == 2ULL);
    CHECK(nanovec_wal_checkpoint(&wal, 2ULL) == NANOVEC_SUCCESS);

    CHECK(nanovec_wal_log_update(&wal, 2ULL, v2u, sizeof(v2u), &lsn) == NANOVEC_SUCCESS && lsn == 3ULL);
    CHECK(nanovec_wal_log_delete(&wal, 1ULL, &lsn) == NANOVEC_SUCCESS && lsn == 4ULL);
    CHECK(nanovec_wal_log_metadata(&wal, 2ULL, m3, (uint32_t)strlen(m3), &lsn) == NANOVEC_SUCCESS && lsn == 5ULL);

    uint8_t *big = (uint8_t *)malloc(10000U);
    CHECK(big != NULL);
    for (size_t i = 0U; i < 10000U; ++i) {
        big[i] = (uint8_t)((i * 37U + 5U) & 0xFFU);
    }
    CHECK(nanovec_wal_log_insert(&wal, 15ULL, big, 10000U, &lsn) == NANOVEC_SUCCESS && lsn == 6ULL);
    uint64_t clean_end = wal.write_offset;
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* 3. Simulate crash with a torn partial record header (13 bytes) at EOF. */
    nanovec_file_t rawf;
    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    const uint8_t torn_hdr[13] = {0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x09};
    CHECK(nanovec_file_pwrite_exact(&rawf, clean_end, torn_hdr, sizeof(torn_hdr)) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);

    /* 4. Callback error aborts recovery WITHOUT truncating the WAL so retry still works. */
    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    replay_state_t rstate;
    memset(&rstate, 0, sizeof(rstate));
    rstate.fail_on_call = 2U;
    CHECK(nanovec_wal_recover(&wal, test_replay_cb, &rstate, &stats) == NANOVEC_ERR_OOM);
    uint64_t fsz = 0ULL;
    CHECK(nanovec_file_size(&wal.file, &fsz) == NANOVEC_SUCCESS);
    CHECK(fsz == clean_end + sizeof(torn_hdr));

    /* 5. Retry recovery: replays LSNs 3..6, skips 1..2, and truncates the 13B torn tail. */
    memset(&rstate, 0, sizeof(rstate));
    rstate.present[1] = true; /* pre-checkpoint state */
    rstate.present[2] = true;
    CHECK(nanovec_wal_recover(&wal, test_replay_cb, &rstate, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 6ULL);
    CHECK(stats.records_skipped == 2ULL);
    CHECK(stats.records_replayed == 4ULL);
    CHECK(stats.last_valid_lsn == 6ULL);
    CHECK(stats.valid_end_offset == clean_end);
    CHECK(stats.had_torn_tail && stats.truncated_bytes == sizeof(torn_hdr));

    CHECK(!rstate.present[1]);
    CHECK(rstate.present[2]);
    CHECK(memcmp(rstate.vecs[2], v2u, sizeof(v2u)) == 0);
    CHECK(strcmp(rstate.meta[2], m3) == 0);
    CHECK(rstate.big_len == 10000U && rstate.big_first == big[0] && rstate.big_last == big[9999]);
    free(big);

    CHECK(nanovec_file_size(&wal.file, &fsz) == NANOVEC_SUCCESS);
    CHECK(fsz == clean_end);
    CHECK(wal.write_offset == clean_end && wal.next_lsn == 7ULL);

    /* Second recovery pass on now-repaired WAL reports clean tail. */
    CHECK(nanovec_wal_recover(&wal, NULL, NULL, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 6ULL && stats.records_skipped == 2ULL && stats.records_replayed == 0ULL);
    CHECK(!stats.had_torn_tail && stats.truncated_bytes == 0ULL);

    /* 6. Append record 7, then corrupt its payload byte (CRC-32C mismatch) and recover. */
    uint64_t before_rec7 = wal.write_offset;
    CHECK(nanovec_wal_log_insert(&wal, 3ULL, v1, sizeof(v1), &lsn) == NANOVEC_SUCCESS && lsn == 7ULL);
    uint64_t after_rec7 = wal.write_offset;
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    uint8_t flip = 0xAAU;
    CHECK(nanovec_file_pwrite_exact(&rawf, before_rec7 + NANOVEC_WAL_RECORD_HDR_SZ + 4ULL, &flip, 1U) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);

    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_recover(&wal, NULL, NULL, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 6ULL && stats.last_valid_lsn == 6ULL);
    CHECK(stats.had_torn_tail && stats.truncated_bytes == (after_rec7 - before_rec7));
    CHECK(wal.write_offset == before_rec7 && wal.next_lsn == 7ULL);

    /* 7. Append a new valid record 7, then simulate torn payload (truncate mid-payload). */
    CHECK(nanovec_wal_log_insert(&wal, 4ULL, v2, sizeof(v2), &lsn) == NANOVEC_SUCCESS && lsn == 7ULL);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    CHECK(nanovec_file_open(path, 0U, &rawf) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_truncate(&rawf, before_rec7 + NANOVEC_WAL_RECORD_HDR_SZ + 5ULL) == NANOVEC_SUCCESS);
    CHECK(nanovec_file_close(&rawf) == NANOVEC_SUCCESS);

    CHECK(nanovec_wal_open(path, true, &wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_recover(&wal, NULL, NULL, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 6ULL && stats.had_torn_tail);
    CHECK(stats.truncated_bytes == (uint64_t)NANOVEC_WAL_RECORD_HDR_SZ + 5ULL);
    CHECK(wal.write_offset == before_rec7 && wal.next_lsn == 7ULL);

    /* New append after recovery succeeds and persists cleanly. */
    CHECK(nanovec_wal_log_insert(&wal, 5ULL, v1, sizeof(v1), &lsn) == NANOVEC_SUCCESS && lsn == 7ULL);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    CHECK(nanovec_wal_open(path, false, &wal) == NANOVEC_SUCCESS);
    CHECK(nanovec_wal_recover(&wal, NULL, NULL, &stats) == NANOVEC_SUCCESS);
    CHECK(stats.records_scanned == 7ULL && stats.last_valid_lsn == 7ULL && !stats.had_torn_tail);
    CHECK(nanovec_wal_close(&wal) == NANOVEC_SUCCESS);

    /* Defensive error checks. */
    CHECK(nanovec_wal_recover(NULL, NULL, NULL, &stats) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_wal_recover(&wal, NULL, NULL, &stats) == NANOVEC_ERR_INVALID_ARG);

    CHECK(unlink(path) == 0);
}

typedef struct growth_reader_ctx {
    nanovec_growable_map_t *gmap;
    _Atomic bool           *stop;
    size_t                  vec_off;
    size_t                  initial_bytes;
    uint64_t                reads_done;
} growth_reader_ctx_t;

static void *growth_reader_fn(void *arg) {
    growth_reader_ctx_t *ctx = (growth_reader_ctx_t *)arg;
    uint64_t prev_gen = 0ULL;
    while (!atomic_load_explicit(ctx->stop, memory_order_acquire)) {
        nanovec_growth_ptrs_t p;
        CHECK(nanovec_growth_read_lock(ctx->gmap, &p) == NANOVEC_SUCCESS);
        CHECK(p.base != NULL && p.header != NULL && p.vectors != NULL);
        CHECK(p.vectors == (void *)(p.base + ctx->vec_off));
        CHECK(p.generation >= prev_gen);
        prev_gen = p.generation;

        const uint8_t *v = (const uint8_t *)p.vectors;
        for (size_t i = 0U; i < ctx->initial_bytes; i += 64U) {
            CHECK(v[i] == (uint8_t)((i * 29U + 7U) & 0xFFU));
        }
        nanovec_growth_read_unlock(ctx->gmap);
        ctx->reads_done++;
    }
    return NULL;
}

static void test_dynamic_growth_mremap(void) {
    char path[256];
    make_temp_path(path, sizeof(path), "growth");
    (void)unlink(path);

    long   ps_raw    = sysconf(_SC_PAGESIZE);
    size_t page_size = (ps_raw > 0) ? (size_t)ps_raw : NANOVEC_PAGE_SIZE;

    /* Verify geometric growth size calculation. */
    CHECK(nanovec_growth_calc_size(page_size, page_size, page_size) == page_size);
    CHECK(nanovec_growth_calc_size(page_size, page_size + 1U, page_size) == 2U * page_size);
    CHECK(nanovec_growth_calc_size(page_size, 5U * page_size, page_size) == 8U * page_size);

    nanovec_file_header_t init_hdr;
    nanovec_header_init(&init_hdr, 32U, NANOVEC_METRIC_L2, NANOVEC_INDEX_HNSW, NANOVEC_QUANT_NONE, 16ULL);
    init_hdr.vectors_offset  = nanovec_htole64((uint64_t)page_size);
    init_hdr.metadata_offset = nanovec_htole64(2ULL * (uint64_t)page_size);

    nanovec_mmap_t map;
    size_t init_size = 3U * page_size;
    CHECK(nanovec_mmap_open_nvec(path, &init_hdr, init_size, false, true, &map) == NANOVEC_SUCCESS);

    uint8_t *vecs = (uint8_t *)nanovec_mmap_vectors_ptr(&map);
    CHECK(vecs != NULL);
    for (size_t i = 0U; i < page_size; ++i) {
        vecs[i] = (uint8_t)((i * 29U + 7U) & 0xFFU);
    }

    nanovec_growable_map_t gmap;
    CHECK(nanovec_growth_init(&gmap, map) == NANOVEC_SUCCESS);

    nanovec_growth_ptrs_t p0;
    CHECK(nanovec_growth_read_lock(&gmap, &p0) == NANOVEC_SUCCESS);
    CHECK(p0.mapped_len == init_size);
    CHECK(p0.header != NULL && p0.vectors == (void *)(p0.base + page_size));
    CHECK(p0.metadata == (void *)(p0.base + 2U * page_size));
    CHECK(p0.generation == 1ULL);
    nanovec_growth_read_unlock(&gmap);

    /* Ensure capacity within current size is a no-op. */
    bool grew = true;
    CHECK(nanovec_growth_ensure_capacity(&gmap, init_size, &grew) == NANOVEC_SUCCESS);
    CHECK(!grew);

    /* Spawn 3 concurrent reader threads while writer performs multiple mremap growths up to 4 MiB. */
    _Atomic bool stop_flag = false;
    pthread_t           readers[3];
    growth_reader_ctx_t rctxs[3];
    for (int t = 0; t < 3; ++t) {
        rctxs[t].gmap          = &gmap;
        rctxs[t].stop          = &stop_flag;
        rctxs[t].vec_off       = page_size;
        rctxs[t].initial_bytes = page_size;
        rctxs[t].reads_done    = 0ULL;
        CHECK(pthread_create(&readers[t], NULL, growth_reader_fn, &rctxs[t]) == 0);
    }

    const size_t targets[4] = {
        64U * 1024U,
        256U * 1024U,
        1024U * 1024U,
        4U * 1024U * 1024U
    };
    for (int i = 0; i < 4; ++i) {
        grew = false;
        CHECK(nanovec_growth_ensure_capacity(&gmap, targets[i], &grew) == NANOVEC_SUCCESS);
        CHECK(grew);

        nanovec_growth_ptrs_t pi;
        CHECK(nanovec_growth_read_lock(&gmap, &pi) == NANOVEC_SUCCESS);
        CHECK(pi.mapped_len >= targets[i]);
        CHECK(pi.header != NULL && nanovec_header_verify(pi.header) == NANOVEC_SUCCESS);
        CHECK(pi.vectors == (void *)(pi.base + page_size));
        CHECK(pi.metadata == (void *)(pi.base + 2U * page_size));
        /* Touch the newly grown tail page to confirm no SIGBUS and zero-extension. */
        CHECK(pi.base[pi.mapped_len - 1U] == 0U);
        pi.base[pi.mapped_len - 1U] = (uint8_t)(0xA0 + i);
        nanovec_growth_read_unlock(&gmap);
    }

    /* Exact growth while readers are active, then join readers before updating header offsets. */
    size_t exact_target = gmap.map.mapped_len + 2U * page_size;
    CHECK(nanovec_growth_grow_exact(&gmap, exact_target, NULL) == NANOVEC_SUCCESS);

    atomic_store_explicit(&stop_flag, true, memory_order_release);
    for (int t = 0; t < 3; ++t) {
        CHECK(pthread_join(readers[t], NULL) == 0);
        CHECK(rctxs[t].reads_done > 0ULL);
    }

    gmap.ptrs.header->graph_offset = nanovec_htole64((uint64_t)(exact_target - page_size));
    nanovec_header_seal(gmap.ptrs.header);
    CHECK(nanovec_growth_refresh_ptrs(&gmap) == NANOVEC_SUCCESS);

    nanovec_growth_ptrs_t pf;
    CHECK(nanovec_growth_read_lock(&gmap, &pf) == NANOVEC_SUCCESS);
    CHECK(pf.mapped_len == exact_target);
    CHECK(pf.graph == (void *)(pf.base + exact_target - page_size));
    CHECK(nanovec_header_verify(pf.header) == NANOVEC_SUCCESS);
    nanovec_growth_read_unlock(&gmap);

    CHECK(nanovec_mmap_sync(&gmap.map, false) == NANOVEC_SUCCESS);
    CHECK(nanovec_growth_destroy(&gmap) == NANOVEC_SUCCESS);

    /* Reopen read-only: verify persisted grown size and rejection of growth on read-only map. */
    CHECK(nanovec_mmap_open_nvec(path, NULL, 0U, true, false, &map) == NANOVEC_SUCCESS);
    CHECK(map.mapped_len == exact_target);
    CHECK(nanovec_mmap_grow(&map, exact_target + page_size, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_close(&map) == NANOVEC_SUCCESS);

    /* Defensive error checks. */
    CHECK(nanovec_mmap_grow(NULL, 4096U, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_growth_ensure_capacity(NULL, 4096U, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_growth_read_lock(NULL, &pf) == NANOVEC_ERR_INVALID_ARG);

    CHECK(unlink(path) == 0);
}

typedef struct seqlock_reader_ctx {
    const nanovec_mmap_header_state_t *state;
    _Atomic bool                      *stop;
    uint64_t                           reads;
} seqlock_reader_ctx_t;

static void *seqlock_reader_fn(void *arg) {
    seqlock_reader_ctx_t *ctx      = (seqlock_reader_ctx_t *)arg;
    uint64_t              prev_k   = 0ULL;
    uint64_t              prev_seq = 0ULL;

    while (!atomic_load_explicit(ctx->stop, memory_order_acquire)) {
        nanovec_header_snapshot_t snap;
        CHECK(nanovec_mmap_header_read_snapshot(ctx->state, &snap) == NANOVEC_SUCCESS);
        CHECK((snap.sequence & 1ULL) == 0ULL);
        CHECK(snap.sequence >= prev_seq);
        prev_seq = snap.sequence;

        uint64_t k = snap.vector_count;
        CHECK(k >= prev_k);
        prev_k = k;

        /* Verify all 6 fields obey the exact atomic invariant (zero torn reads). */
        CHECK(snap.max_capacity == k * 2ULL + 16ULL);
        CHECK(snap.vectors_offset == 128ULL + k * 64ULL);
        CHECK(snap.quant_offset == 128ULL + k * 128ULL);
        CHECK(snap.graph_offset == 128ULL + k * 192ULL);
        CHECK(snap.metadata_offset == 128ULL + k * 256ULL);
        ctx->reads++;
    }
    return NULL;
}

static void test_seqlock_header_concurrency(void) {
    /* 1. Raw seqlock primitives: even -> odd -> even & retry detection. */
    nanovec_seqlock_t sl;
    nanovec_seqlock_init(&sl);
    uint64_t s0 = nanovec_seqlock_read_begin(&sl);
    CHECK(s0 == 0ULL);
    CHECK(!nanovec_seqlock_read_retry(&sl, s0));

    nanovec_seqlock_write_begin(&sl);
    CHECK((atomic_load_explicit(&sl.seq, memory_order_relaxed) & 1ULL) == 1ULL);
    nanovec_seqlock_write_end(&sl);
    CHECK(nanovec_seqlock_read_retry(&sl, s0));

    uint64_t s2 = nanovec_seqlock_read_begin(&sl);
    CHECK(s2 == 2ULL);
    CHECK(!nanovec_seqlock_read_retry(&sl, s2));

    /* 2. .nvec mmap-backed header state initialization and publish + CRC verify. */
    char path[256];
    make_temp_path(path, sizeof(path), "seqlock");
    (void)unlink(path);

    nanovec_file_header_t init_hdr;
    nanovec_header_init(&init_hdr, 32U, NANOVEC_METRIC_L2, NANOVEC_INDEX_FLAT, NANOVEC_QUANT_NONE, 16ULL);
    init_hdr.vectors_offset  = nanovec_htole64(128ULL);
    init_hdr.quant_offset    = nanovec_htole64(128ULL);
    init_hdr.graph_offset    = nanovec_htole64(128ULL);
    init_hdr.metadata_offset = nanovec_htole64(128ULL);

    nanovec_mmap_t map;
    CHECK(nanovec_mmap_open_nvec(path, &init_hdr, 4096U, false, true, &map) == NANOVEC_SUCCESS);

    nanovec_mmap_header_state_t state;
    CHECK(nanovec_mmap_header_state_init(&state, &map) == NANOVEC_SUCCESS);

    nanovec_header_snapshot_t snap;
    CHECK(nanovec_mmap_header_read_snapshot(&state, &snap) == NANOVEC_SUCCESS);
    CHECK(snap.vector_count == 0ULL && snap.max_capacity == 16ULL);
    CHECK(snap.vectors_offset == 128ULL && snap.metadata_offset == 128ULL);

    /* 3. Concurrent 1-writer + 4-reader tear-free stress test (20,000 updates). */
    _Atomic bool stop = false;
    enum { NREADERS = 4, NUPDATES = 20000 };
    pthread_t            readers[NREADERS];
    seqlock_reader_ctx_t rctxs[NREADERS];

    for (int i = 0; i < NREADERS; ++i) {
        rctxs[i].state = &state;
        rctxs[i].stop  = &stop;
        rctxs[i].reads = 0ULL;
        CHECK(pthread_create(&readers[i], NULL, seqlock_reader_fn, &rctxs[i]) == 0);
    }

    for (uint64_t k = 1ULL; k <= NUPDATES; ++k) {
        nanovec_header_snapshot_t next;
        next.vector_count    = k;
        next.max_capacity    = k * 2ULL + 16ULL;
        next.vectors_offset  = 128ULL + k * 64ULL;
        next.quant_offset    = 128ULL + k * 128ULL;
        next.graph_offset    = 128ULL + k * 192ULL;
        next.metadata_offset = 128ULL + k * 256ULL;
        next.sequence        = 0ULL;
        /* Persist header updates to mmap every 1000 steps + final step. */
        nanovec_mmap_t *target_map = ((k % 1000ULL) == 0ULL || k == NUPDATES) ? &map : NULL;
        CHECK(nanovec_mmap_header_publish(&state, target_map, &next) == NANOVEC_SUCCESS);
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    uint64_t total_reads = 0ULL;
    for (int i = 0; i < NREADERS; ++i) {
        CHECK(pthread_join(readers[i], NULL) == 0);
        CHECK(rctxs[i].reads > 0ULL);
        total_reads += rctxs[i].reads;
    }
    CHECK(total_reads >= 4ULL);

    /* Final snapshot and on-disk .nvec header must match and pass CRC-32C. */
    CHECK(nanovec_mmap_header_read_snapshot(&state, &snap) == NANOVEC_SUCCESS);
    CHECK(snap.vector_count == NUPDATES && snap.max_capacity == NUPDATES * 2ULL + 16ULL);
    CHECK(snap.sequence == (uint64_t)NUPDATES * 2ULL);
    CHECK(nanovec_header_verify(nanovec_mmap_header(&map)) == NANOVEC_SUCCESS);
    CHECK(nanovec_le64toh(nanovec_mmap_header(&map)->vector_count) == NUPDATES);

    /* publish_count fast path updates count/capacity and preserves offsets. */
    CHECK(nanovec_mmap_header_publish_count(&state, &map, 424242ULL, 500000ULL) == NANOVEC_SUCCESS);
    CHECK(nanovec_mmap_header_read_snapshot(&state, &snap) == NANOVEC_SUCCESS);
    CHECK(snap.vector_count == 424242ULL && snap.max_capacity == 500000ULL);
    CHECK(snap.vectors_offset == 128ULL + NUPDATES * 64ULL);
    CHECK(nanovec_header_verify(nanovec_mmap_header(&map)) == NANOVEC_SUCCESS);

    /* Defensive error checks. */
    CHECK(nanovec_mmap_header_state_init(NULL, &map) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_header_read_snapshot(NULL, &snap) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_header_read_snapshot(&state, NULL) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_header_publish(&state, &map, NULL) == NANOVEC_ERR_INVALID_ARG);

    nanovec_mmap_header_state_destroy(&state);
    CHECK(nanovec_mmap_header_read_snapshot(&state, &snap) == NANOVEC_ERR_INVALID_ARG);
    CHECK(nanovec_mmap_close(&map) == NANOVEC_SUCCESS);
    CHECK(unlink(path) == 0);
}

int main(void) {
    test_header_and_checksum();
    test_file_io();
    test_mmap_and_madvise();
    test_madvise_small_file();
    test_wal_append_and_reopen();
    test_wal_concurrent_appends();
    test_wal_recover_and_crash_replay();
    test_dynamic_growth_mremap();
    test_seqlock_header_concurrency();
    (void)printf("test_storage: ALL PASSED\n");
    return 0;
}
