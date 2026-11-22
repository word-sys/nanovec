#ifndef NANOVEC_CORE_ALLOC_H
#define NANOVEC_CORE_ALLOC_H

#include "core/types.h"

void *nanovec_aligned_alloc(size_t alignment, size_t size);
void *nanovec_aligned_calloc(size_t alignment, size_t count, size_t elem_size);
void *nanovec_aligned_realloc(void *ptr, size_t old_size, size_t new_size, size_t alignment);
void  nanovec_aligned_free(void *ptr);

#endif /* NANOVEC_CORE_ALLOC_H */
