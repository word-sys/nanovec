#include "core/cpu.h"

#include <pthread.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#elif (defined(__aarch64__) || defined(__arm__)) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

nanovec_cpu_caps_t g_nanovec_cpu_caps;
static pthread_once_t g_cpu_once = PTHREAD_ONCE_INIT;

#if defined(__x86_64__) || defined(__i386__)
static uint64_t nanovec_xgetbv0(void) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0U));
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}
#endif

static void nanovec_cpu_detect_once(void) {
    nanovec_cpu_caps_t caps;
    memset(&caps, 0, sizeof(caps));

#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = __get_cpuid_max(0U, NULL);

    bool os_avx    = false;
    bool os_avx512 = false;

    if (max_leaf >= 1U && __get_cpuid(1U, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1U << 20)) {
            caps.sse42 = 1U;
            caps.crc32 = 1U;
            caps.mask |= NANOVEC_CPU_CAP_SSE42 | NANOVEC_CPU_CAP_CRC32;
        }
        if (ecx & (1U << 23)) {
            caps.popcnt = 1U;
            caps.mask  |= NANOVEC_CPU_CAP_POPCNT;
        }

        bool has_osxsave = (ecx & (1U << 27)) != 0U;
        if (has_osxsave) {
            uint64_t xcr0 = nanovec_xgetbv0();
            os_avx    = (xcr0 & 0x06ULL) == 0x06ULL;
            os_avx512 = os_avx && ((xcr0 & 0xE0ULL) == 0xE0ULL);
        }

        if (os_avx && (ecx & (1U << 28))) {
            caps.avx   = 1U;
            caps.mask |= NANOVEC_CPU_CAP_AVX;
        }
        if (os_avx && (ecx & (1U << 12))) {
            caps.fma   = 1U;
            caps.mask |= NANOVEC_CPU_CAP_FMA;
        }
        if (os_avx && (ecx & (1U << 29))) {
            caps.f16c  = 1U;
            caps.mask |= NANOVEC_CPU_CAP_F16C;
        }
    }

    if (max_leaf >= 7U && __get_cpuid_count(7U, 0U, &eax, &ebx, &ecx, &edx)) {
        if (os_avx && (ebx & (1U << 5))) {
            caps.avx2  = 1U;
            caps.mask |= NANOVEC_CPU_CAP_AVX2;
        }
        if (os_avx512) {
            if (ebx & (1U << 16)) {
                caps.avx512f = 1U;
                caps.mask   |= NANOVEC_CPU_CAP_AVX512F;
            }
            if (ebx & (1U << 17)) {
                caps.avx512dq = 1U;
                caps.mask    |= NANOVEC_CPU_CAP_AVX512DQ;
            }
            if (ebx & (1U << 30)) {
                caps.avx512bw = 1U;
                caps.mask    |= NANOVEC_CPU_CAP_AVX512BW;
            }
            if (ebx & (1U << 31)) {
                caps.avx512vl = 1U;
                caps.mask    |= NANOVEC_CPU_CAP_AVX512VL;
            }
        }
    }
#elif defined(__aarch64__) && defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
#ifdef HWCAP_ASIMD
    if (hwcap & HWCAP_ASIMD) {
        caps.neon  = 1U;
        caps.asimd = 1U;
        caps.mask |= NANOVEC_CPU_CAP_NEON | NANOVEC_CPU_CAP_ASIMD;
    }
#endif
#ifdef HWCAP_CRC32
    if (hwcap & HWCAP_CRC32) {
        caps.crc32 = 1U;
        caps.mask |= NANOVEC_CPU_CAP_CRC32;
    }
#endif
#ifdef HWCAP_ASIMDDP
    if (hwcap & HWCAP_ASIMDDP) {
        caps.dotprod = 1U;
        caps.mask   |= NANOVEC_CPU_CAP_DOTPROD;
    }
#endif
#elif defined(__arm__) && defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
#ifdef HWCAP_NEON
    if (hwcap & HWCAP_NEON) {
        caps.neon  = 1U;
        caps.mask |= NANOVEC_CPU_CAP_NEON;
    }
#endif
#endif

    g_nanovec_cpu_caps = caps;
}

void nanovec_cpu_detect(void) {
    (void)pthread_once(&g_cpu_once, nanovec_cpu_detect_once);
}

const nanovec_cpu_caps_t *nanovec_cpu_get_caps(void) {
    nanovec_cpu_detect();
    return &g_nanovec_cpu_caps;
}

uint32_t nanovec_cpu_caps_mask(void) {
    return nanovec_cpu_get_caps()->mask;
}

const char *nanovec_cpu_best_isa_name(void) {
    const nanovec_cpu_caps_t *c = nanovec_cpu_get_caps();
    if (c->avx512f && c->avx512dq && c->avx512bw) {
        return "avx512";
    }
    if (c->avx2 && c->fma) {
        return "avx2";
    }
    if (c->neon || c->asimd) {
        return "neon";
    }
    return "scalar";
}

uint32_t nanovec_cpu_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1U;
}
