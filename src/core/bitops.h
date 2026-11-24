/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_CORE_BITOPS_H
#define NANOVEC_CORE_BITOPS_H

#include "core/types.h"

static inline uint32_t nanovec_popcount64_scalar(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555ULL;
    x  = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x  = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (uint32_t)((x * 0x0101010101010101ULL) >> 56);
}

static inline uint32_t nanovec_clz64_scalar(uint64_t x) {
    if (x == 0) {
        return 64U;
    }
    uint32_t n = 0;
    if ((x & 0xFFFFFFFF00000000ULL) == 0) { n += 32U; x <<= 32; }
    if ((x & 0xFFFF000000000000ULL) == 0) { n += 16U; x <<= 16; }
    if ((x & 0xFF00000000000000ULL) == 0) { n += 8U;  x <<= 8;  }
    if ((x & 0xF000000000000000ULL) == 0) { n += 4U;  x <<= 4;  }
    if ((x & 0xC000000000000000ULL) == 0) { n += 2U;  x <<= 2;  }
    if ((x & 0x8000000000000000ULL) == 0) { n += 1U; }
    return n;
}

static inline uint32_t nanovec_ctz64_scalar(uint64_t x) {
    if (x == 0) {
        return 64U;
    }
    uint32_t n = 0;
    if ((x & 0x00000000FFFFFFFFULL) == 0) { n += 32U; x >>= 32; }
    if ((x & 0x000000000000FFFFULL) == 0) { n += 16U; x >>= 16; }
    if ((x & 0x00000000000000FFULL) == 0) { n += 8U;  x >>= 8;  }
    if ((x & 0x000000000000000FULL) == 0) { n += 4U;  x >>= 4;  }
    if ((x & 0x0000000000000003ULL) == 0) { n += 2U;  x >>= 2;  }
    if ((x & 0x0000000000000001ULL) == 0) { n += 1U; }
    return n;
}

static inline uint32_t nanovec_popcount64(uint64_t x) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_popcountll((unsigned long long)x);
#else
    return nanovec_popcount64_scalar(x);
#endif
}

static inline uint32_t nanovec_clz64(uint64_t x) {
    if (x == 0) {
        return 64U;
    }
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_clzll((unsigned long long)x);
#else
    return nanovec_clz64_scalar(x);
#endif
}

static inline uint32_t nanovec_ctz64(uint64_t x) {
    if (x == 0) {
        return 64U;
    }
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_ctzll((unsigned long long)x);
#else
    return nanovec_ctz64_scalar(x);
#endif
}

static inline uint32_t nanovec_popcount32(uint32_t x) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_popcount(x);
#else
    return nanovec_popcount64_scalar((uint64_t)x);
#endif
}

static inline uint32_t nanovec_clz32(uint32_t x) {
    if (x == 0) {
        return 32U;
    }
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_clz(x);
#else
    return nanovec_clz64_scalar((uint64_t)x) - 32U;
#endif
}

static inline uint32_t nanovec_ctz32(uint32_t x) {
    if (x == 0) {
        return 32U;
    }
#if (defined(__GNUC__) || defined(__clang__)) && !defined(NANOVEC_FORCE_SCALAR_BITOPS)
    return (uint32_t)__builtin_ctz(x);
#else
    return nanovec_ctz64_scalar((uint64_t)x);
#endif
}

static inline uint64_t nanovec_rotl64(uint64_t x, unsigned int r) {
    r &= 63U;
    return (x << r) | (x >> ((64U - r) & 63U));
}

static inline uint64_t nanovec_rotr64(uint64_t x, unsigned int r) {
    r &= 63U;
    return (x >> r) | (x << ((64U - r) & 63U));
}

static inline uint32_t nanovec_rotl32(uint32_t x, unsigned int r) {
    r &= 31U;
    return (x << r) | (x >> ((32U - r) & 31U));
}

static inline uint32_t nanovec_rotr32(uint32_t x, unsigned int r) {
    r &= 31U;
    return (x >> r) | (x << ((32U - r) & 31U));
}

static inline uint64_t nanovec_next_pow2_u64(uint64_t x) {
    if (x <= 1ULL) {
        return 1ULL;
    }
    if (x > (1ULL << 63)) {
        return 0ULL;
    }
    return 1ULL << (64U - nanovec_clz64(x - 1ULL));
}

static inline size_t nanovec_bitset_words(size_t num_bits) {
    return (num_bits + 63U) >> 6;
}

static inline bool nanovec_bitset_test(const uint64_t *bits, size_t idx) {
    return (bits[idx >> 6] & (1ULL << (idx & 63U))) != 0ULL;
}

static inline void nanovec_bitset_set(uint64_t *bits, size_t idx) {
    bits[idx >> 6] |= (1ULL << (idx & 63U));
}

static inline void nanovec_bitset_clear(uint64_t *bits, size_t idx) {
    bits[idx >> 6] &= ~(1ULL << (idx & 63U));
}

#endif /* NANOVEC_CORE_BITOPS_H */
