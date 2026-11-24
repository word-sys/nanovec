#ifndef NANOVEC_MATH_PREFETCH_H
#define NANOVEC_MATH_PREFETCH_H

#include "core/types.h"

#if !defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#include <xmmintrin.h>
#endif

static inline void nanovec_prefetch_l1(const void *ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_prefetch((const char *)ptr, _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

static inline void nanovec_prefetch_l2(const void *ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 2);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_prefetch((const char *)ptr, _MM_HINT_T1);
#else
    (void)ptr;
#endif
}

static inline void nanovec_prefetch_l3(const void *ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 1);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_prefetch((const char *)ptr, _MM_HINT_T2);
#else
    (void)ptr;
#endif
}

static inline void nanovec_prefetch_nta(const void *ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 0);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_prefetch((const char *)ptr, _MM_HINT_NTA);
#else
    (void)ptr;
#endif
}

static inline void nanovec_prefetch_write(const void *ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 1, 3);
#else
    (void)ptr;
#endif
}

static inline void nanovec_prefetch_range_l1(const void *ptr, size_t bytes) {
    if (NANOVEC_UNLIKELY(ptr == NULL || bytes == 0)) {
        return;
    }
    const uint8_t *p = (const uint8_t *)ptr;
    for (size_t off = 0; off < bytes; off += NANOVEC_CACHE_LINE) {
        nanovec_prefetch_l1(p + off);
    }
}

static inline void nanovec_prefetch_range_l2(const void *ptr, size_t bytes) {
    if (NANOVEC_UNLIKELY(ptr == NULL || bytes == 0)) {
        return;
    }
    const uint8_t *p = (const uint8_t *)ptr;
    for (size_t off = 0; off < bytes; off += NANOVEC_CACHE_LINE) {
        nanovec_prefetch_l2(p + off);
    }
}

#endif /* NANOVEC_MATH_PREFETCH_H */
