#ifndef NANOVEC_CORE_ALLOC_H
#define NANOVEC_CORE_ALLOC_H

#include "core/types.h"

void *nanovec_aligned_alloc(size_t alignment, size_t size);
void *nanovec_aligned_calloc(size_t alignment, size_t count, size_t elem_size);
void *nanovec_aligned_realloc(void *ptr, size_t old_size, size_t new_size, size_t alignment);
void  nanovec_aligned_free(void *ptr);

typedef struct nanovec_slab_chunk nanovec_slab_chunk_t;

typedef struct nanovec_slab {
    nanovec_slab_chunk_t *chunks;
    void                 *free_list;
    uint8_t              *bump_cur;
    uint8_t              *bump_end;
    size_t                slot_size;
    size_t                slot_align;
    size_t                header_size;
    size_t                slots_per_chunk;
    size_t                chunk_count;
    size_t                active_count;
} nanovec_slab_t;

nanovec_status_t nanovec_slab_init(
    nanovec_slab_t *slab,
    size_t          slot_size,
    size_t          slot_align,
    size_t          slots_per_chunk
);
void  *nanovec_slab_alloc(nanovec_slab_t *slab);
void  *nanovec_slab_calloc(nanovec_slab_t *slab);
void   nanovec_slab_free(nanovec_slab_t *slab, void *ptr);
void   nanovec_slab_reset(nanovec_slab_t *slab);
void   nanovec_slab_destroy(nanovec_slab_t *slab);
size_t nanovec_slab_active_count(const nanovec_slab_t *slab);
size_t nanovec_slab_capacity(const nanovec_slab_t *slab);

typedef struct nanovec_arena_block nanovec_arena_block_t;

typedef struct nanovec_arena {
    nanovec_arena_block_t *head;
    nanovec_arena_block_t *current;
    size_t                 block_size;
    size_t                 total_used;
    size_t                 total_capacity;
    size_t                 block_count;
} nanovec_arena_t;

typedef struct nanovec_arena_mark {
    nanovec_arena_block_t *block;
    size_t                 used;
    size_t                 total_used;
} nanovec_arena_mark_t;

nanovec_status_t     nanovec_arena_init(nanovec_arena_t *arena, size_t initial_capacity);
void                *nanovec_arena_alloc(nanovec_arena_t *arena, size_t size, size_t align);
void                *nanovec_arena_calloc(
    nanovec_arena_t *arena,
    size_t           count,
    size_t           elem_size,
    size_t           align
);
nanovec_arena_mark_t nanovec_arena_save(const nanovec_arena_t *arena);
void                 nanovec_arena_restore(nanovec_arena_t *arena, nanovec_arena_mark_t mark);
void                 nanovec_arena_reset(nanovec_arena_t *arena);
void                 nanovec_arena_trim(nanovec_arena_t *arena);
void                 nanovec_arena_destroy(nanovec_arena_t *arena);
size_t               nanovec_arena_used(const nanovec_arena_t *arena);
size_t               nanovec_arena_capacity(const nanovec_arena_t *arena);

#endif /* NANOVEC_CORE_ALLOC_H */
