/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_CORE_TYPES_H
#define NANOVEC_CORE_TYPES_H

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__linux__)
#include <endian.h>
#endif

#include "nanovec.h"

typedef uint64_t nanovec_vid_t;
typedef float    nanovec_dist_t;
typedef uint16_t nanovec_dim_t;
typedef uint32_t nanovec_idx_t;

#define NANOVEC_VID_INVALID UINT64_MAX
#define NANOVEC_IDX_INVALID UINT32_MAX

#define NANOVEC_CACHE_LINE 64U
#define NANOVEC_PAGE_SIZE  4096U

#define NANOVEC_ALIGNAS(n) _Alignas(n)
#define NANOVEC_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#define NANOVEC_ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define NANOVEC_RESTRICT restrict

#if defined(__GNUC__) || defined(__clang__)
#define NANOVEC_LIKELY(x)   __builtin_expect(!!(x), 1)
#define NANOVEC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define NANOVEC_NOINLINE    __attribute__((noinline))
#define NANOVEC_PACKED      __attribute__((packed))
#define NANOVEC_UNUSED      __attribute__((unused))
#else
#define NANOVEC_LIKELY(x)   (x)
#define NANOVEC_UNLIKELY(x) (x)
#define NANOVEC_NOINLINE
#define NANOVEC_PACKED
#define NANOVEC_UNUSED
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define NANOVEC_LITTLE_ENDIAN 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define NANOVEC_LITTLE_ENDIAN 0
#else
#error "Unsupported byte order"
#endif
#elif defined(__BYTE_ORDER) && defined(__LITTLE_ENDIAN) && defined(__BIG_ENDIAN)
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define NANOVEC_LITTLE_ENDIAN 1
#elif __BYTE_ORDER == __BIG_ENDIAN
#define NANOVEC_LITTLE_ENDIAN 0
#else
#error "Unsupported byte order"
#endif
#else
#error "Unable to determine host endianness"
#endif

static inline uint16_t nanovec_bswap16(uint16_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap16(v);
#else
    return (uint16_t)((v >> 8) | (v << 8));
#endif
}

static inline uint32_t nanovec_bswap32(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(v);
#else
    return ((v & 0x000000FFU) << 24) |
           ((v & 0x0000FF00U) << 8)  |
           ((v & 0x00FF0000U) >> 8)  |
           ((v & 0xFF000000U) >> 24);
#endif
}

static inline uint64_t nanovec_bswap64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    return ((v & 0x00000000000000FFULL) << 56) |
           ((v & 0x000000000000FF00ULL) << 40) |
           ((v & 0x0000000000FF0000ULL) << 24) |
           ((v & 0x00000000FF000000ULL) << 8)  |
           ((v & 0x000000FF00000000ULL) >> 8)  |
           ((v & 0x0000FF0000000000ULL) >> 24) |
           ((v & 0x00FF000000000000ULL) >> 40) |
           ((v & 0xFF00000000000000ULL) >> 56);
#endif
}

static inline uint16_t nanovec_htole16(uint16_t v) {
#if NANOVEC_LITTLE_ENDIAN
    return v;
#else
    return nanovec_bswap16(v);
#endif
}

static inline uint32_t nanovec_htole32(uint32_t v) {
#if NANOVEC_LITTLE_ENDIAN
    return v;
#else
    return nanovec_bswap32(v);
#endif
}

static inline uint64_t nanovec_htole64(uint64_t v) {
#if NANOVEC_LITTLE_ENDIAN
    return v;
#else
    return nanovec_bswap64(v);
#endif
}

static inline uint16_t nanovec_le16toh(uint16_t v) { return nanovec_htole16(v); }
static inline uint32_t nanovec_le32toh(uint32_t v) { return nanovec_htole32(v); }
static inline uint64_t nanovec_le64toh(uint64_t v) { return nanovec_htole64(v); }

#ifndef htole16
#define htole16(x) nanovec_htole16((uint16_t)(x))
#endif
#ifndef htole32
#define htole32(x) nanovec_htole32((uint32_t)(x))
#endif
#ifndef htole64
#define htole64(x) nanovec_htole64((uint64_t)(x))
#endif
#ifndef le16toh
#define le16toh(x) nanovec_le16toh((uint16_t)(x))
#endif
#ifndef le32toh
#define le32toh(x) nanovec_le32toh((uint32_t)(x))
#endif
#ifndef le64toh
#define le64toh(x) nanovec_le64toh((uint64_t)(x))
#endif
#ifndef letoh16
#define letoh16(x) le16toh(x)
#endif
#ifndef letoh32
#define letoh32(x) le32toh(x)
#endif
#ifndef letoh64
#define letoh64(x) le64toh(x)
#endif

static inline uint32_t nanovec_f32_to_le(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return nanovec_htole32(bits);
}

static inline float nanovec_le_to_f32(uint32_t u) {
    uint32_t bits = nanovec_le32toh(u);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t nanovec_read_le16(const void *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return nanovec_le16toh(v);
}

static inline uint32_t nanovec_read_le32(const void *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return nanovec_le32toh(v);
}

static inline uint64_t nanovec_read_le64(const void *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return nanovec_le64toh(v);
}

static inline float nanovec_read_f32_le(const void *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return nanovec_le_to_f32(v);
}

static inline void nanovec_write_le16(void *p, uint16_t v) {
    uint16_t le = nanovec_htole16(v);
    memcpy(p, &le, sizeof(le));
}

static inline void nanovec_write_le32(void *p, uint32_t v) {
    uint32_t le = nanovec_htole32(v);
    memcpy(p, &le, sizeof(le));
}

static inline void nanovec_write_le64(void *p, uint64_t v) {
    uint64_t le = nanovec_htole64(v);
    memcpy(p, &le, sizeof(le));
}

static inline void nanovec_write_f32_le(void *p, float v) {
    uint32_t le = nanovec_f32_to_le(v);
    memcpy(p, &le, sizeof(le));
}

static inline bool nanovec_is_pow2(size_t x) {
    return x != 0 && (x & (x - 1U)) == 0;
}

static inline size_t nanovec_align_up(size_t val, size_t align) {
    return (val + (align - 1U)) & ~(align - 1U);
}

#endif /* NANOVEC_CORE_TYPES_H */
