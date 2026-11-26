/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/checksum.h"
#include "core/bitops.h"
#include "core/cpu.h"
#include "core/error.h"

#include <pthread.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

#define NANOVEC_CRC32C_POLY 0x82F63B78U

static uint32_t       g_crc32c_table[256];
static pthread_once_t g_crc32c_once = PTHREAD_ONCE_INIT;

static void nanovec_crc32c_init_table(void) {
    for (uint32_t i = 0; i < 256U; ++i) {
        uint32_t c = i;
        for (int b = 0; b < 8; ++b) {
            c = (c & 1U) ? (NANOVEC_CRC32C_POLY ^ (c >> 1)) : (c >> 1);
        }
        g_crc32c_table[i] = c;
    }
}

uint32_t nanovec_crc32c_scalar(uint32_t crc, const void *data, size_t len) {
    if (NANOVEC_UNLIKELY(data == NULL || len == 0U)) {
        return crc;
    }
    (void)pthread_once(&g_crc32c_once, nanovec_crc32c_init_table);

    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; ++i) {
        c = g_crc32c_table[(c ^ p[i]) & 0xFFU] ^ (c >> 8);
    }
    return ~c;
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse4.2")))
static uint32_t nanovec_crc32c_sse42(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    size_t i = 0;

#if defined(__x86_64__)
    uint64_t c = (uint64_t)(~crc);
    while (i + 8U <= len) {
        uint64_t v;
        memcpy(&v, p + i, sizeof(v));
        c = _mm_crc32_u64(c, v);
        i += 8U;
    }
    uint32_t c32 = (uint32_t)c;
#else
    uint32_t c32 = ~crc;
    while (i + 4U <= len) {
        uint32_t v;
        memcpy(&v, p + i, sizeof(v));
        c32 = _mm_crc32_u32(c32, v);
        i += 4U;
    }
#endif
    while (i < len) {
        c32 = _mm_crc32_u8(c32, p[i]);
        ++i;
    }
    return ~c32;
}
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
static uint32_t nanovec_crc32c_arm(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = ~crc;
    size_t i = 0;

    while (i + 8U <= len) {
        uint64_t v;
        memcpy(&v, p + i, sizeof(v));
        c = __crc32cd(c, v);
        i += 8U;
    }
    while (i < len) {
        c = __crc32cb(c, p[i]);
        ++i;
    }
    return ~c;
}
#endif

uint32_t nanovec_crc32c_update(uint32_t crc, const void *data, size_t len) {
    if (NANOVEC_UNLIKELY(data == NULL || len == 0U)) {
        return crc;
    }

    const nanovec_cpu_caps_t *caps = nanovec_cpu_get_caps();
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (caps->crc32) {
        return nanovec_crc32c_sse42(crc, data, len);
    }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    if (caps->crc32) {
        return nanovec_crc32c_arm(crc, data, len);
    }
#else
    (void)caps;
#endif
    return nanovec_crc32c_scalar(crc, data, len);
}

uint32_t nanovec_crc32c(const void *data, size_t len) {
    return nanovec_crc32c_update(0U, data, len);
}

#define NANOVEC_XXH64_P1 0x9E3779B185EBCA87ULL
#define NANOVEC_XXH64_P2 0xC2B2AE3D27D4EB4FULL
#define NANOVEC_XXH64_P3 0x165667B19E3779F9ULL
#define NANOVEC_XXH64_P4 0x85EBCA77C2B2AE63ULL
#define NANOVEC_XXH64_P5 0x27D4EB2F165667C5ULL

static inline uint64_t nanovec_xxh64_round(uint64_t acc, uint64_t lane) {
    acc += lane * NANOVEC_XXH64_P2;
    acc  = nanovec_rotl64(acc, 31U);
    return acc * NANOVEC_XXH64_P1;
}

static inline uint64_t nanovec_xxh64_merge_round(uint64_t acc, uint64_t val) {
    val  = nanovec_xxh64_round(0ULL, val);
    acc ^= val;
    return acc * NANOVEC_XXH64_P1 + NANOVEC_XXH64_P4;
}

uint64_t nanovec_xxh64(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p   = (const uint8_t *)data;
    const uint8_t *end = (p != NULL) ? (p + len) : NULL;
    uint64_t h64;

    if (p != NULL && len >= 32U) {
        const uint8_t *limit = end - 32U;
        uint64_t v1 = seed + NANOVEC_XXH64_P1 + NANOVEC_XXH64_P2;
        uint64_t v2 = seed + NANOVEC_XXH64_P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - NANOVEC_XXH64_P1;

        do {
            v1 = nanovec_xxh64_round(v1, nanovec_read_le64(p));      p += 8;
            v2 = nanovec_xxh64_round(v2, nanovec_read_le64(p));      p += 8;
            v3 = nanovec_xxh64_round(v3, nanovec_read_le64(p));      p += 8;
            v4 = nanovec_xxh64_round(v4, nanovec_read_le64(p));      p += 8;
        } while (p <= limit);

        h64 = nanovec_rotl64(v1, 1U) + nanovec_rotl64(v2, 7U)
            + nanovec_rotl64(v3, 12U) + nanovec_rotl64(v4, 18U);
        h64 = nanovec_xxh64_merge_round(h64, v1);
        h64 = nanovec_xxh64_merge_round(h64, v2);
        h64 = nanovec_xxh64_merge_round(h64, v3);
        h64 = nanovec_xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + NANOVEC_XXH64_P5;
    }

    h64 += (uint64_t)len;

    if (p != NULL) {
        while (p + 8 <= end) {
            uint64_t k1 = nanovec_xxh64_round(0ULL, nanovec_read_le64(p));
            h64 ^= k1;
            h64  = nanovec_rotl64(h64, 27U) * NANOVEC_XXH64_P1 + NANOVEC_XXH64_P4;
            p += 8;
        }
        if (p + 4 <= end) {
            h64 ^= (uint64_t)nanovec_read_le32(p) * NANOVEC_XXH64_P1;
            h64  = nanovec_rotl64(h64, 23U) * NANOVEC_XXH64_P2 + NANOVEC_XXH64_P3;
            p += 4;
        }
        while (p < end) {
            h64 ^= (uint64_t)(*p) * NANOVEC_XXH64_P5;
            h64  = nanovec_rotl64(h64, 11U) * NANOVEC_XXH64_P1;
            ++p;
        }
    }

    h64 ^= h64 >> 33;
    h64 *= NANOVEC_XXH64_P2;
    h64 ^= h64 >> 29;
    h64 *= NANOVEC_XXH64_P3;
    h64 ^= h64 >> 32;
    return h64;
}

uint32_t nanovec_header_compute_crc32(const nanovec_file_header_t *hdr) {
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return 0U;
    }
    return nanovec_crc32c(hdr, NANOVEC_HEADER_CRC_RANGE_LEN);
}

void nanovec_header_seal(nanovec_file_header_t *hdr) {
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        return;
    }
    hdr->header_crc32 = nanovec_htole32(nanovec_header_compute_crc32(hdr));
    hdr->crc32_pad    = 0U;
}

nanovec_status_t nanovec_header_verify(const nanovec_file_header_t *hdr) {
    nanovec_status_t st = nanovec_header_validate_basic(hdr);
    if (NANOVEC_UNLIKELY(st != NANOVEC_SUCCESS)) {
        return (nanovec_status_t)nanovec_set_error(st, "nvec header validation failed");
    }

    uint32_t expected = nanovec_le32toh(hdr->header_crc32);
    uint32_t actual   = nanovec_header_compute_crc32(hdr);
    if (NANOVEC_UNLIKELY(expected != actual)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "nvec header CRC-32C mismatch (stored=0x%08x computed=0x%08x)",
            expected,
            actual
        );
    }
    return NANOVEC_SUCCESS;
}
