#include "core/alloc.h"
#include "core/error.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void nanovec_assert_aligned64(const void *ptr) {
    if (NANOVEC_UNLIKELY(((uintptr_t)ptr & (NANOVEC_CACHE_LINE - 1U)) != 0)) {
        (void)fprintf(
            stderr,
            "nanovec: fatal alignment violation: ptr=%p is not 64-byte aligned\n",
            ptr
        );
        abort();
    }
}

static size_t nanovec_resolve_alignment(size_t alignment) {
    if (alignment == 0) {
        return NANOVEC_CACHE_LINE;
    }
    if (!nanovec_is_pow2(alignment)) {
        return 0;
    }
    return alignment < NANOVEC_CACHE_LINE ? NANOVEC_CACHE_LINE : alignment;
}

void *nanovec_aligned_alloc(size_t alignment, size_t size) {
    size_t eff_align = nanovec_resolve_alignment(alignment);
    if (NANOVEC_UNLIKELY(eff_align == 0)) {
        (void)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "alignment %zu must be a power of two",
            alignment
        );
        return NULL;
    }
    if (NANOVEC_UNLIKELY(size == 0)) {
        (void)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "allocation size must be greater than zero"
        );
        return NULL;
    }

    void *ptr = NULL;
    int rc = posix_memalign(&ptr, eff_align, size);
    if (NANOVEC_UNLIKELY(rc != 0 || ptr == NULL)) {
        int code = (rc == EINVAL) ? NANOVEC_ERR_INVALID_ARG : NANOVEC_ERR_OOM;
        (void)nanovec_set_errno(
            code,
            rc,
            "posix_memalign(align=%zu, size=%zu) failed",
            eff_align,
            size
        );
        return NULL;
    }

    nanovec_assert_aligned64(ptr);
    return ptr;
}

void *nanovec_aligned_calloc(size_t alignment, size_t count, size_t elem_size) {
    if (NANOVEC_UNLIKELY(count == 0 || elem_size == 0)) {
        (void)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "calloc count (%zu) and elem_size (%zu) must be greater than zero",
            count,
            elem_size
        );
        return NULL;
    }
    if (NANOVEC_UNLIKELY(elem_size > SIZE_MAX / count)) {
        (void)nanovec_set_error(
            NANOVEC_ERR_OOM,
            "calloc size overflow (%zu * %zu)",
            count,
            elem_size
        );
        return NULL;
    }

    size_t total = count * elem_size;
    void *ptr = nanovec_aligned_alloc(alignment, total);
    if (ptr != NULL) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void *nanovec_aligned_realloc(void *ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (ptr == NULL) {
        return nanovec_aligned_alloc(alignment, new_size);
    }
    nanovec_assert_aligned64(ptr);

    if (new_size == 0) {
        nanovec_aligned_free(ptr);
        return NULL;
    }

    void *next = nanovec_aligned_alloc(alignment, new_size);
    if (next == NULL) {
        return NULL;
    }

    size_t copy_len = old_size < new_size ? old_size : new_size;
    if (copy_len > 0) {
        memcpy(next, ptr, copy_len);
    }
    nanovec_aligned_free(ptr);
    return next;
}

void nanovec_aligned_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    nanovec_assert_aligned64(ptr);
    free(ptr);
}
