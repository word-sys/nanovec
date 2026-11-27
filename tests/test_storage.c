/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/checksum.h"
#include "storage/file.h"
#include "storage/format.h"
#include "storage/mmap.h"

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

int main(void) {
    test_header_and_checksum();
    test_file_io();
    test_mmap_and_madvise();
    test_madvise_small_file();
    (void)printf("test_storage: ALL PASSED\n");
    return 0;
}
