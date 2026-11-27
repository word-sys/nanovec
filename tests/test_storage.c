/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/checksum.h"
#include "storage/file.h"
#include "storage/format.h"
#include "storage/mmap.h"
#include "storage/wal.h"

#include <pthread.h>
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

int main(void) {
    test_header_and_checksum();
    test_file_io();
    test_mmap_and_madvise();
    test_madvise_small_file();
    test_wal_append_and_reopen();
    test_wal_concurrent_appends();
    test_wal_recover_and_crash_replay();
    (void)printf("test_storage: ALL PASSED\n");
    return 0;
}
