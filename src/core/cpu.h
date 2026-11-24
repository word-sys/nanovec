/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_CORE_CPU_H
#define NANOVEC_CORE_CPU_H

#include "core/types.h"

#define NANOVEC_CPU_CAP_SSE42    (1U << 0)
#define NANOVEC_CPU_CAP_POPCNT   (1U << 1)
#define NANOVEC_CPU_CAP_FMA      (1U << 2)
#define NANOVEC_CPU_CAP_AVX      (1U << 3)
#define NANOVEC_CPU_CAP_AVX2     (1U << 4)
#define NANOVEC_CPU_CAP_AVX512F  (1U << 5)
#define NANOVEC_CPU_CAP_AVX512DQ (1U << 6)
#define NANOVEC_CPU_CAP_AVX512BW (1U << 7)
#define NANOVEC_CPU_CAP_AVX512VL (1U << 8)
#define NANOVEC_CPU_CAP_F16C     (1U << 9)
#define NANOVEC_CPU_CAP_NEON     (1U << 10)
#define NANOVEC_CPU_CAP_ASIMD    (1U << 11)
#define NANOVEC_CPU_CAP_DOTPROD  (1U << 12)
#define NANOVEC_CPU_CAP_CRC32    (1U << 13)

typedef struct nanovec_cpu_caps {
    uint32_t mask;
    uint32_t sse42     : 1;
    uint32_t popcnt    : 1;
    uint32_t fma       : 1;
    uint32_t avx       : 1;
    uint32_t avx2      : 1;
    uint32_t avx512f   : 1;
    uint32_t avx512dq  : 1;
    uint32_t avx512bw  : 1;
    uint32_t avx512vl  : 1;
    uint32_t f16c      : 1;
    uint32_t neon      : 1;
    uint32_t asimd     : 1;
    uint32_t dotprod   : 1;
    uint32_t crc32     : 1;
    uint32_t _reserved : 18;
} nanovec_cpu_caps_t;

extern nanovec_cpu_caps_t g_nanovec_cpu_caps;

void                      nanovec_cpu_detect(void);
const nanovec_cpu_caps_t *nanovec_cpu_get_caps(void);
uint32_t                  nanovec_cpu_caps_mask(void);
const char               *nanovec_cpu_best_isa_name(void);
uint32_t                  nanovec_cpu_logical_cores(void);

#endif /* NANOVEC_CORE_CPU_H */
