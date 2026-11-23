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

struct nanovec_slab_chunk {
    nanovec_slab_chunk_t *next;
    size_t                capacity;
};

static nanovec_status_t nanovec_slab_grow(nanovec_slab_t *slab) {
    size_t chunk_align = slab->slot_align > NANOVEC_CACHE_LINE
                             ? slab->slot_align
                             : NANOVEC_CACHE_LINE;
    size_t payload_bytes = slab->slots_per_chunk * slab->slot_size;
    size_t total_bytes   = slab->header_size + payload_bytes;

    uint8_t *raw = (uint8_t *)nanovec_aligned_alloc(chunk_align, total_bytes);
    if (NANOVEC_UNLIKELY(raw == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    nanovec_slab_chunk_t *chunk = (nanovec_slab_chunk_t *)raw;
    chunk->next     = slab->chunks;
    chunk->capacity = slab->slots_per_chunk;

    slab->chunks   = chunk;
    slab->bump_cur = raw + slab->header_size;
    slab->bump_end = slab->bump_cur + payload_bytes;
    slab->chunk_count++;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_slab_init(
    nanovec_slab_t *slab,
    size_t          slot_size,
    size_t          slot_align,
    size_t          slots_per_chunk
) {
    if (NANOVEC_UNLIKELY(slab == NULL || slot_size == 0 || slots_per_chunk == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "slab_init requires non-null slab, slot_size > 0, slots_per_chunk > 0"
        );
    }
    if (NANOVEC_UNLIKELY(slot_align != 0 && !nanovec_is_pow2(slot_align))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "slab slot_align %zu must be a power of two",
            slot_align
        );
    }

    memset(slab, 0, sizeof(*slab));

    size_t min_align = _Alignof(void *);
    size_t eff_align = slot_align == 0 ? min_align
                                       : (slot_align < min_align ? min_align : slot_align);
    size_t min_slot  = slot_size < sizeof(void *) ? sizeof(void *) : slot_size;
    if (NANOVEC_UNLIKELY(min_slot > SIZE_MAX - eff_align)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_OOM, "slab slot_size overflow");
    }
    size_t eff_slot = nanovec_align_up(min_slot, eff_align);

    size_t hdr_align = eff_align > NANOVEC_CACHE_LINE ? eff_align : NANOVEC_CACHE_LINE;
    size_t hdr_size  = nanovec_align_up(sizeof(nanovec_slab_chunk_t), hdr_align);

    if (NANOVEC_UNLIKELY(slots_per_chunk > (SIZE_MAX - hdr_size) / eff_slot)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_OOM, "slab chunk size overflow");
    }

    slab->slot_size       = eff_slot;
    slab->slot_align      = eff_align;
    slab->header_size     = hdr_size;
    slab->slots_per_chunk = slots_per_chunk;

    nanovec_status_t st = nanovec_slab_grow(slab);
    if (st != NANOVEC_SUCCESS) {
        memset(slab, 0, sizeof(*slab));
        return st;
    }
    return NANOVEC_SUCCESS;
}

void *nanovec_slab_alloc(nanovec_slab_t *slab) {
    if (NANOVEC_UNLIKELY(slab == NULL || slab->slot_size == 0)) {
        (void)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "uninitialized slab");
        return NULL;
    }

    if (slab->free_list != NULL) {
        void *slot = slab->free_list;
        void *next = NULL;
        memcpy(&next, slot, sizeof(next));
        slab->free_list = next;
        slab->active_count++;
        return slot;
    }

    if (NANOVEC_UNLIKELY(slab->bump_cur >= slab->bump_end)) {
        if (nanovec_slab_grow(slab) != NANOVEC_SUCCESS) {
            return NULL;
        }
    }

    void *slot = slab->bump_cur;
    slab->bump_cur += slab->slot_size;
    slab->active_count++;
    return slot;
}

void *nanovec_slab_calloc(nanovec_slab_t *slab) {
    void *slot = nanovec_slab_alloc(slab);
    if (slot != NULL) {
        memset(slot, 0, slab->slot_size);
    }
    return slot;
}

void nanovec_slab_free(nanovec_slab_t *slab, void *ptr) {
    if (NANOVEC_UNLIKELY(slab == NULL || ptr == NULL)) {
        return;
    }
    if (NANOVEC_UNLIKELY(((uintptr_t)ptr & (slab->slot_align - 1U)) != 0)) {
        (void)fprintf(
            stderr,
            "nanovec: fatal slab alignment violation: ptr=%p align=%zu\n",
            ptr,
            slab->slot_align
        );
        abort();
    }
    if (NANOVEC_UNLIKELY(slab->active_count == 0)) {
        (void)fprintf(stderr, "nanovec: fatal slab double-free detected\n");
        abort();
    }

    void *head = slab->free_list;
    memcpy(ptr, &head, sizeof(head));
    slab->free_list = ptr;
    slab->active_count--;
}

void nanovec_slab_reset(nanovec_slab_t *slab) {
    if (NANOVEC_UNLIKELY(slab == NULL || slab->chunks == NULL)) {
        return;
    }

    nanovec_slab_chunk_t *keep = slab->chunks;
    nanovec_slab_chunk_t *cur  = keep->next;
    while (cur != NULL) {
        nanovec_slab_chunk_t *next = cur->next;
        nanovec_aligned_free(cur);
        cur = next;
    }

    keep->next        = NULL;
    slab->chunks      = keep;
    slab->free_list   = NULL;
    slab->bump_cur    = (uint8_t *)keep + slab->header_size;
    slab->bump_end    = slab->bump_cur + (keep->capacity * slab->slot_size);
    slab->chunk_count = 1U;
    slab->active_count = 0U;
}

void nanovec_slab_destroy(nanovec_slab_t *slab) {
    if (slab == NULL) {
        return;
    }

    nanovec_slab_chunk_t *cur = slab->chunks;
    while (cur != NULL) {
        nanovec_slab_chunk_t *next = cur->next;
        nanovec_aligned_free(cur);
        cur = next;
    }
    memset(slab, 0, sizeof(*slab));
}

size_t nanovec_slab_active_count(const nanovec_slab_t *slab) {
    return slab ? slab->active_count : 0U;
}

size_t nanovec_slab_capacity(const nanovec_slab_t *slab) {
    return slab ? slab->chunk_count * slab->slots_per_chunk : 0U;
}

struct nanovec_arena_block {
    nanovec_arena_block_t *next;
    uint8_t               *data;
    size_t                 capacity;
    size_t                 used;
};

static nanovec_arena_block_t *nanovec_arena_block_new(size_t capacity, size_t align) {
    size_t eff_align = align > NANOVEC_CACHE_LINE ? align : NANOVEC_CACHE_LINE;
    size_t hdr_size  = nanovec_align_up(sizeof(nanovec_arena_block_t), eff_align);

    if (NANOVEC_UNLIKELY(capacity > SIZE_MAX - eff_align || capacity > SIZE_MAX - hdr_size)) {
        (void)nanovec_set_error(NANOVEC_ERR_OOM, "arena block size overflow");
        return NULL;
    }

    size_t payload_cap = nanovec_align_up(capacity, NANOVEC_CACHE_LINE);
    if (NANOVEC_UNLIKELY(payload_cap > SIZE_MAX - hdr_size)) {
        (void)nanovec_set_error(NANOVEC_ERR_OOM, "arena block total size overflow");
        return NULL;
    }

    uint8_t *raw = (uint8_t *)nanovec_aligned_alloc(eff_align, hdr_size + payload_cap);
    if (NANOVEC_UNLIKELY(raw == NULL)) {
        return NULL;
    }

    nanovec_arena_block_t *blk = (nanovec_arena_block_t *)raw;
    blk->next     = NULL;
    blk->data     = raw + hdr_size;
    blk->capacity = payload_cap;
    blk->used     = 0U;
    return blk;
}

static void *nanovec_arena_block_try_alloc(
    nanovec_arena_block_t *blk,
    size_t                 size,
    size_t                 align,
    size_t                *out_consumed
) {
    uintptr_t cur = (uintptr_t)(blk->data + blk->used);
    if (NANOVEC_UNLIKELY(cur > UINTPTR_MAX - (align - 1U))) {
        return NULL;
    }
    uintptr_t aligned = (cur + (align - 1U)) & ~(align - 1U);
    size_t    pad     = (size_t)(aligned - cur);
    size_t    avail   = blk->capacity - blk->used;

    if (pad > avail || size > avail - pad) {
        return NULL;
    }

    size_t consumed = pad + size;
    blk->used += consumed;
    *out_consumed = consumed;
    return (void *)aligned;
}

nanovec_status_t nanovec_arena_init(nanovec_arena_t *arena, size_t initial_capacity) {
    if (NANOVEC_UNLIKELY(arena == NULL || initial_capacity == 0)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "arena_init requires non-null arena and initial_capacity > 0"
        );
    }

    memset(arena, 0, sizeof(*arena));

    nanovec_arena_block_t *blk = nanovec_arena_block_new(initial_capacity, NANOVEC_CACHE_LINE);
    if (blk == NULL) {
        return NANOVEC_ERR_OOM;
    }

    arena->head           = blk;
    arena->current        = blk;
    arena->block_size     = blk->capacity;
    arena->total_used     = 0U;
    arena->total_capacity = blk->capacity;
    arena->block_count    = 1U;
    return NANOVEC_SUCCESS;
}

void *nanovec_arena_alloc(nanovec_arena_t *arena, size_t size, size_t align) {
    if (NANOVEC_UNLIKELY(arena == NULL || arena->current == NULL || size == 0)) {
        (void)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid arena_alloc arguments");
        return NULL;
    }
    if (NANOVEC_UNLIKELY(align != 0 && !nanovec_is_pow2(align))) {
        (void)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "arena align %zu must be a power of two",
            align
        );
        return NULL;
    }

    size_t eff_align = align == 0 ? _Alignof(max_align_t) : align;
    size_t consumed  = 0U;

    void *ptr = nanovec_arena_block_try_alloc(arena->current, size, eff_align, &consumed);
    if (NANOVEC_LIKELY(ptr != NULL)) {
        arena->total_used += consumed;
        return ptr;
    }

    nanovec_arena_block_t *blk = arena->current;
    while (blk->next != NULL) {
        blk = blk->next;
        blk->used = 0U;
        ptr = nanovec_arena_block_try_alloc(blk, size, eff_align, &consumed);
        if (ptr != NULL) {
            arena->current     = blk;
            arena->total_used += consumed;
            return ptr;
        }
    }

    size_t next_cap = size > arena->block_size ? size : arena->block_size;
    nanovec_arena_block_t *fresh = nanovec_arena_block_new(next_cap, eff_align);
    if (NANOVEC_UNLIKELY(fresh == NULL)) {
        return NULL;
    }

    blk->next              = fresh;
    arena->current         = fresh;
    arena->total_capacity += fresh->capacity;
    arena->block_count++;

    ptr = nanovec_arena_block_try_alloc(fresh, size, eff_align, &consumed);
    arena->total_used += consumed;
    return ptr;
}

void *nanovec_arena_calloc(
    nanovec_arena_t *arena,
    size_t           count,
    size_t           elem_size,
    size_t           align
) {
    if (NANOVEC_UNLIKELY(count == 0 || elem_size == 0)) {
        (void)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "arena_calloc count (%zu) and elem_size (%zu) must be > 0",
            count,
            elem_size
        );
        return NULL;
    }
    if (NANOVEC_UNLIKELY(elem_size > SIZE_MAX / count)) {
        (void)nanovec_set_error(NANOVEC_ERR_OOM, "arena_calloc size overflow");
        return NULL;
    }

    size_t total = count * elem_size;
    void *ptr = nanovec_arena_alloc(arena, total, align);
    if (ptr != NULL) {
        memset(ptr, 0, total);
    }
    return ptr;
}

nanovec_arena_mark_t nanovec_arena_save(const nanovec_arena_t *arena) {
    nanovec_arena_mark_t mark = { NULL, 0U, 0U };
    if (arena != NULL && arena->current != NULL) {
        mark.block      = arena->current;
        mark.used       = arena->current->used;
        mark.total_used = arena->total_used;
    }
    return mark;
}

void nanovec_arena_restore(nanovec_arena_t *arena, nanovec_arena_mark_t mark) {
    if (NANOVEC_UNLIKELY(arena == NULL || mark.block == NULL)) {
        return;
    }

    mark.block->used = mark.used;
    nanovec_arena_block_t *b = mark.block->next;
    while (b != NULL) {
        b->used = 0U;
        b = b->next;
    }
    arena->current    = mark.block;
    arena->total_used = mark.total_used;
}

void nanovec_arena_reset(nanovec_arena_t *arena) {
    if (NANOVEC_UNLIKELY(arena == NULL || arena->head == NULL)) {
        return;
    }

    nanovec_arena_block_t *b = arena->head;
    while (b != NULL) {
        b->used = 0U;
        b = b->next;
    }
    arena->current    = arena->head;
    arena->total_used = 0U;
}

void nanovec_arena_trim(nanovec_arena_t *arena) {
    if (NANOVEC_UNLIKELY(arena == NULL || arena->head == NULL)) {
        return;
    }

    nanovec_arena_block_t *b = arena->head->next;
    while (b != NULL) {
        nanovec_arena_block_t *next = b->next;
        nanovec_aligned_free(b);
        b = next;
    }

    arena->head->next     = NULL;
    arena->head->used     = 0U;
    arena->current        = arena->head;
    arena->total_used     = 0U;
    arena->total_capacity = arena->head->capacity;
    arena->block_count    = 1U;
}

void nanovec_arena_destroy(nanovec_arena_t *arena) {
    if (arena == NULL) {
        return;
    }

    nanovec_arena_block_t *b = arena->head;
    while (b != NULL) {
        nanovec_arena_block_t *next = b->next;
        nanovec_aligned_free(b);
        b = next;
    }
    memset(arena, 0, sizeof(*arena));
}

size_t nanovec_arena_used(const nanovec_arena_t *arena) {
    return arena ? arena->total_used : 0U;
}

size_t nanovec_arena_capacity(const nanovec_arena_t *arena) {
    return arena ? arena->total_capacity : 0U;
}
