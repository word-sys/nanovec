/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "storage/metadata.h"
#include "core/alloc.h"
#include "core/error.h"
#include "storage/checksum.h"

#include <string.h>

#define NANOVEC_META_MIN_SLOTS       64U
#define NANOVEC_META_DEFAULT_BYTES   4096U
#define NANOVEC_META_AUTO_COMPACT_SZ (64U * 1024U)

enum {
    NANOVEC_META_SLOT_EMPTY     = 0,
    NANOVEC_META_SLOT_OCCUPIED  = 1,
    NANOVEC_META_SLOT_TOMBSTONE = 2
};

static inline uint64_t nanovec_meta_hash_u64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline const uint8_t *nanovec_meta_base_const(const nanovec_metadata_store_t *store) {
    return store->readonly ? store->mmap_data : store->heap_data;
}

static nanovec_status_t nanovec_meta_rdlock(const nanovec_metadata_store_t *store) {
    if (NANOVEC_UNLIKELY(store == NULL || !store->initialized || !store->lock_init)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid metadata store handle"
        );
    }
    int rc = pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock rdlock failed");
    }
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_wrlock(nanovec_metadata_store_t *store) {
    if (NANOVEC_UNLIKELY(store == NULL || !store->initialized || !store->lock_init)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid metadata store handle"
        );
    }
    int rc = pthread_rwlock_wrlock(&store->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock wrlock failed");
    }
    return NANOVEC_SUCCESS;
}

static void nanovec_meta_unlock(const nanovec_metadata_store_t *store) {
    (void)pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
}

static nanovec_meta_slot_t *nanovec_meta_find_slot(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id
) {
    if (store->slot_cap == 0U || store->slots == NULL) {
        return NULL;
    }
    size_t   mask = store->slot_cap - 1U;
    size_t   idx  = (size_t)nanovec_meta_hash_u64(id) & mask;
    size_t   step = 0U;

    while (step < store->slot_cap) {
        nanovec_meta_slot_t *s = &store->slots[idx];
        if (s->state == NANOVEC_META_SLOT_EMPTY) {
            return NULL;
        }
        if (s->state == NANOVEC_META_SLOT_OCCUPIED && s->id == id) {
            return s;
        }
        idx = (idx + 1U) & mask;
        step++;
    }
    return NULL;
}

static nanovec_status_t nanovec_meta_rehash(nanovec_metadata_store_t *store, size_t new_cap) {
    if (new_cap < NANOVEC_META_MIN_SLOTS) {
        new_cap = NANOVEC_META_MIN_SLOTS;
    }
    if (!nanovec_is_pow2(new_cap)) {
        new_cap = nanovec_align_up(new_cap, NANOVEC_META_MIN_SLOTS);
        size_t p = NANOVEC_META_MIN_SLOTS;
        while (p < new_cap && p <= SIZE_MAX / 2U) {
            p *= 2U;
        }
        new_cap = p;
    }

    nanovec_meta_slot_t *next = (nanovec_meta_slot_t *)nanovec_aligned_calloc(
        NANOVEC_CACHE_LINE, new_cap, sizeof(nanovec_meta_slot_t)
    );
    if (NANOVEC_UNLIKELY(next == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    size_t mask = new_cap - 1U;
    for (size_t i = 0U; i < store->slot_cap; ++i) {
        const nanovec_meta_slot_t *src = &store->slots[i];
        if (src->state != NANOVEC_META_SLOT_OCCUPIED) {
            continue;
        }
        size_t idx = (size_t)nanovec_meta_hash_u64(src->id) & mask;
        while (next[idx].state == NANOVEC_META_SLOT_OCCUPIED) {
            idx = (idx + 1U) & mask;
        }
        next[idx] = *src;
    }

    nanovec_aligned_free(store->slots);
    store->slots           = next;
    store->slot_cap        = new_cap;
    store->tombstone_count = 0U;
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_maybe_grow_table(nanovec_metadata_store_t *store) {
    size_t used = store->live_count + store->tombstone_count;
    if ((used + 1U) * 10U < store->slot_cap * 7U) {
        return NANOVEC_SUCCESS;
    }
    size_t new_cap = store->slot_cap;
    if ((store->live_count + 1U) * 2U >= store->slot_cap) {
        if (NANOVEC_UNLIKELY(store->slot_cap > SIZE_MAX / 2U)) {
            return NANOVEC_ERR_OOM;
        }
        new_cap = store->slot_cap * 2U;
    }
    return nanovec_meta_rehash(store, new_cap);
}

static nanovec_status_t nanovec_meta_upsert_slot(
    nanovec_metadata_store_t *store,
    nanovec_vid_t             id,
    uint64_t                  data_offset,
    uint32_t                  payload_len,
    bool                     *out_replaced,
    uint32_t                 *out_old_len
) {
    nanovec_status_t st = nanovec_meta_maybe_grow_table(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    size_t               mask       = store->slot_cap - 1U;
    size_t               idx        = (size_t)nanovec_meta_hash_u64(id) & mask;
    nanovec_meta_slot_t *first_tomb = NULL;

    for (size_t step = 0U; step < store->slot_cap; ++step) {
        nanovec_meta_slot_t *s = &store->slots[idx];
        if (s->state == NANOVEC_META_SLOT_EMPTY) {
            nanovec_meta_slot_t *dst = (first_tomb != NULL) ? first_tomb : s;
            if (dst->state == NANOVEC_META_SLOT_TOMBSTONE) {
                store->tombstone_count--;
            }
            dst->id          = id;
            dst->data_offset = data_offset;
            dst->payload_len = payload_len;
            dst->state       = NANOVEC_META_SLOT_OCCUPIED;
            store->live_count++;
            if (out_replaced != NULL) *out_replaced = false;
            if (out_old_len != NULL)  *out_old_len  = 0U;
            return NANOVEC_SUCCESS;
        }
        if (s->state == NANOVEC_META_SLOT_TOMBSTONE) {
            if (first_tomb == NULL) {
                first_tomb = s;
            }
        } else if (s->id == id) {
            if (out_replaced != NULL) *out_replaced = true;
            if (out_old_len != NULL)  *out_old_len  = s->payload_len;
            s->data_offset = data_offset;
            s->payload_len = payload_len;
            return NANOVEC_SUCCESS;
        }
        idx = (idx + 1U) & mask;
    }

    return NANOVEC_ERR_INTERNAL;
}

static nanovec_status_t nanovec_meta_ensure_heap_space(
    nanovec_metadata_store_t *store,
    size_t                    extra_bytes
) {
    if (NANOVEC_UNLIKELY(extra_bytes > SIZE_MAX - store->data_used)) {
        return NANOVEC_ERR_OOM;
    }
    size_t needed = store->data_used + extra_bytes;
    if (needed <= store->data_cap) {
        return NANOVEC_SUCCESS;
    }

    size_t new_cap = (store->data_cap > 0U) ? store->data_cap : NANOVEC_META_DEFAULT_BYTES;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2U) {
            new_cap = needed;
            break;
        }
        new_cap *= 2U;
    }
    size_t aligned_cap = nanovec_align_up(new_cap, NANOVEC_CACHE_LINE);
    if (aligned_cap == 0U) {
        return NANOVEC_ERR_OOM;
    }

    uint8_t *next = (uint8_t *)nanovec_aligned_realloc(
        store->heap_data,
        store->data_used,
        aligned_cap,
        NANOVEC_CACHE_LINE
    );
    if (NANOVEC_UNLIKELY(next == NULL)) {
        return NANOVEC_ERR_OOM;
    }
    store->heap_data = next;
    store->data_cap  = aligned_cap;
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_compact_locked(nanovec_metadata_store_t *store) {
    if (store->dead_bytes == 0U) {
        return NANOVEC_SUCCESS;
    }

    size_t live_bytes = store->data_used - store->dead_bytes;
    if (live_bytes == 0U) {
        store->data_used  = 0U;
        store->dead_bytes = 0U;
        return NANOVEC_SUCCESS;
    }

    size_t alloc_sz = nanovec_align_up(live_bytes, NANOVEC_CACHE_LINE);
    if (alloc_sz < NANOVEC_META_DEFAULT_BYTES) {
        alloc_sz = NANOVEC_META_DEFAULT_BYTES;
    }
    uint8_t *compacted = (uint8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, alloc_sz);
    if (NANOVEC_UNLIKELY(compacted == NULL)) {
        return NANOVEC_ERR_OOM;
    }

    size_t pos = 0U;
    for (size_t i = 0U; i < store->slot_cap; ++i) {
        nanovec_meta_slot_t *s = &store->slots[i];
        if (s->state != NANOVEC_META_SLOT_OCCUPIED) {
            continue;
        }
        size_t rec_len = (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)s->payload_len;
        memcpy(compacted + pos, store->heap_data + s->data_offset, rec_len);
        s->data_offset = (uint64_t)pos;
        pos += rec_len;
    }

    nanovec_aligned_free(store->heap_data);
    store->heap_data  = compacted;
    store->data_cap   = alloc_sz;
    store->data_used  = pos;
    store->dead_bytes = 0U;

    if (store->tombstone_count > store->live_count) {
        (void)nanovec_meta_rehash(store, store->slot_cap);
    }
    return NANOVEC_SUCCESS;
}

static size_t nanovec_meta_live_bytes_locked(const nanovec_metadata_store_t *store) {
    if (store->data_used < store->dead_bytes) {
        return 0U;
    }
    return store->data_used - store->dead_bytes;
}

static nanovec_status_t nanovec_meta_serialize_locked(
    const nanovec_metadata_store_t *store,
    void                           *out_buf,
    size_t                          buf_cap,
    size_t                         *out_written
) {
    size_t live_bytes = nanovec_meta_live_bytes_locked(store);
    size_t total_need = (size_t)NANOVEC_META_SECTION_HDR_SZ + live_bytes;
    if (out_written != NULL) {
        *out_written = total_need;
    }
    if (NANOVEC_UNLIKELY(out_buf == NULL || buf_cap < total_need)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "metadata serialize buffer too small"
        );
    }

    const uint8_t *src_base = nanovec_meta_base_const(store);
    uint8_t       *dst      = (uint8_t *)out_buf;
    uint8_t       *rec_dst  = dst + NANOVEC_META_SECTION_HDR_SZ;
    size_t         pos      = 0U;

    for (size_t i = 0U; i < store->slot_cap; ++i) {
        const nanovec_meta_slot_t *s = &store->slots[i];
        if (s->state != NANOVEC_META_SLOT_OCCUPIED) {
            continue;
        }
        size_t rec_sz = (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)s->payload_len;
        if (NANOVEC_UNLIKELY(pos + rec_sz > live_bytes)) {
            return NANOVEC_ERR_CORRUPT;
        }
        memcpy(rec_dst + pos, src_base + s->data_offset, rec_sz);
        pos += rec_sz;
    }
    if (NANOVEC_UNLIKELY(pos != live_bytes)) {
        return NANOVEC_ERR_CORRUPT;
    }

    uint32_t data_crc = (live_bytes > 0U) ? nanovec_crc32c(rec_dst, live_bytes) : 0U;

    nanovec_meta_section_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic[0]     = NANOVEC_META_MAGIC_0;
    hdr.magic[1]     = NANOVEC_META_MAGIC_1;
    hdr.magic[2]     = NANOVEC_META_MAGIC_2;
    hdr.magic[3]     = NANOVEC_META_MAGIC_3;
    hdr.version      = nanovec_htole16((uint16_t)NANOVEC_META_VERSION);
    hdr.flags        = nanovec_htole16(0x0001U);
    hdr.record_count = nanovec_htole64((uint64_t)store->live_count);
    hdr.data_bytes   = nanovec_htole64((uint64_t)live_bytes);
    hdr.data_crc32   = nanovec_htole32(data_crc);
    hdr.header_crc32 = nanovec_htole32(
        nanovec_crc32c(&hdr, offsetof(nanovec_meta_section_header_t, header_crc32))
    );
    memcpy(dst, &hdr, sizeof(hdr));

    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_parse_section(
    const uint8_t                 *buf,
    size_t                         buf_len,
    nanovec_meta_section_header_t *out_hdr
) {
    if (NANOVEC_UNLIKELY(buf == NULL || buf_len < NANOVEC_META_SECTION_HDR_SZ)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "metadata section shorter than 32-byte header"
        );
    }

    nanovec_meta_section_header_t hdr;
    memcpy(&hdr, buf, sizeof(hdr));
    if (NANOVEC_UNLIKELY(
            hdr.magic[0] != NANOVEC_META_MAGIC_0 || hdr.magic[1] != NANOVEC_META_MAGIC_1 ||
            hdr.magic[2] != NANOVEC_META_MAGIC_2 || hdr.magic[3] != NANOVEC_META_MAGIC_3
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "invalid metadata section magic"
        );
    }

    uint16_t ver = nanovec_le16toh(hdr.version);
    if (NANOVEC_UNLIKELY(ver != NANOVEC_META_VERSION)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_VERSION,
            "unsupported metadata section version %u",
            (unsigned)ver
        );
    }

    uint32_t stored_hcrc = nanovec_le32toh(hdr.header_crc32);
    uint32_t calc_hcrc   = nanovec_crc32c(
        &hdr, offsetof(nanovec_meta_section_header_t, header_crc32)
    );
    if (NANOVEC_UNLIKELY(stored_hcrc != calc_hcrc)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "metadata header CRC-32C mismatch"
        );
    }

    uint64_t data_bytes = nanovec_le64toh(hdr.data_bytes);
    if (NANOVEC_UNLIKELY(data_bytes > (uint64_t)(buf_len - NANOVEC_META_SECTION_HDR_SZ))) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "metadata data_bytes exceeds buffer length"
        );
    }

    uint32_t stored_dcrc = nanovec_le32toh(hdr.data_crc32);
    uint32_t calc_dcrc   = (data_bytes > 0ULL)
        ? nanovec_crc32c(buf + NANOVEC_META_SECTION_HDR_SZ, (size_t)data_bytes)
        : 0U;
    if (NANOVEC_UNLIKELY(stored_dcrc != calc_dcrc)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "metadata payload CRC-32C mismatch"
        );
    }

    *out_hdr = hdr;
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_index_records(nanovec_metadata_store_t *store, uint64_t rec_count) {
    const uint8_t *base = nanovec_meta_base_const(store);
    size_t         pos  = 0U;

    for (uint64_t r = 0ULL; r < rec_count; ++r) {
        if (NANOVEC_UNLIKELY(pos + NANOVEC_META_REC_HDR_SZ > store->data_used)) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_CORRUPT,
                "truncated metadata record header"
            );
        }
        uint64_t id  = nanovec_read_le64(base + pos + 0x00);
        uint32_t len = nanovec_read_le32(base + pos + 0x08);
        if (NANOVEC_UNLIKELY(
                len > NANOVEC_META_MAX_PAYLOAD ||
                (size_t)len > store->data_used - (pos + NANOVEC_META_REC_HDR_SZ)
            )) {
            return (nanovec_status_t)nanovec_set_error(
                NANOVEC_ERR_CORRUPT,
                "invalid metadata record length"
            );
        }

        bool     replaced = false;
        uint32_t old_len  = 0U;
        nanovec_status_t st = nanovec_meta_upsert_slot(
            store, (nanovec_vid_t)id, (uint64_t)pos, len, &replaced, &old_len
        );
        if (st != NANOVEC_SUCCESS) {
            return st;
        }
        if (replaced) {
            store->dead_bytes += (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)old_len;
        }
        pos += (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)len;
    }

    if (NANOVEC_UNLIKELY(pos != store->data_used)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_CORRUPT,
            "metadata record byte stream length mismatch"
        );
    }
    return NANOVEC_SUCCESS;
}

static nanovec_status_t nanovec_meta_init_common(
    nanovec_metadata_store_t *store,
    size_t                    slot_cap,
    size_t                    heap_cap,
    bool                      readonly
) {
    memset(store, 0, sizeof(*store));
    store->readonly = readonly;

    pthread_rwlockattr_t attr;
    bool attr_ok = (pthread_rwlockattr_init(&attr) == 0);
#if defined(__GLIBC__) && defined(PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
    if (attr_ok) {
        (void)pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    }
#endif
    int rc = pthread_rwlock_init(&store->rwlock, attr_ok ? &attr : NULL);
    if (attr_ok) {
        (void)pthread_rwlockattr_destroy(&attr);
    }
    if (NANOVEC_UNLIKELY(rc != 0)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INTERNAL, "metadata rwlock init failed");
    }
    store->lock_init = true;

    store->slot_cap = slot_cap;
    store->slots    = (nanovec_meta_slot_t *)nanovec_aligned_calloc(
        NANOVEC_CACHE_LINE, slot_cap, sizeof(nanovec_meta_slot_t)
    );
    if (NANOVEC_UNLIKELY(store->slots == NULL)) {
        (void)pthread_rwlock_destroy(&store->rwlock);
        store->lock_init = false;
        return NANOVEC_ERR_OOM;
    }

    if (!readonly && heap_cap > 0U) {
        size_t aligned = nanovec_align_up(heap_cap, NANOVEC_CACHE_LINE);
        store->heap_data = (uint8_t *)nanovec_aligned_alloc(NANOVEC_CACHE_LINE, aligned);
        if (NANOVEC_UNLIKELY(store->heap_data == NULL)) {
            nanovec_aligned_free(store->slots);
            store->slots = NULL;
            (void)pthread_rwlock_destroy(&store->rwlock);
            store->lock_init = false;
            return NANOVEC_ERR_OOM;
        }
        store->data_cap = aligned;
    }

    store->initialized = true;
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_init(
    nanovec_metadata_store_t *store,
    size_t                    initial_bytes
) {
    if (NANOVEC_UNLIKELY(store == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "store is NULL");
    }
    size_t heap_cap = (initial_bytes > 0U) ? initial_bytes : NANOVEC_META_DEFAULT_BYTES;
    return nanovec_meta_init_common(store, NANOVEC_META_MIN_SLOTS, heap_cap, false);
}

void nanovec_metadata_destroy(nanovec_metadata_store_t *store) {
    if (store == NULL) {
        return;
    }
    nanovec_aligned_free(store->slots);
    nanovec_aligned_free(store->heap_data);
    if (store->lock_init) {
        (void)pthread_rwlock_destroy(&store->rwlock);
        store->lock_init = false;
    }
    memset(store, 0, sizeof(*store));
}

nanovec_status_t nanovec_metadata_put(
    nanovec_metadata_store_t *store,
    nanovec_vid_t             id,
    const void               *payload,
    size_t                    len
) {
    if (NANOVEC_UNLIKELY(
            (len > 0U && payload == NULL) || len > (size_t)NANOVEC_META_MAX_PAYLOAD
        )) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "invalid metadata payload");
    }

    nanovec_status_t st = nanovec_meta_wrlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    if (NANOVEC_UNLIKELY(store->readonly)) {
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "cannot mutate read-only metadata store"
        );
    }

    nanovec_meta_slot_t *existing = nanovec_meta_find_slot(store, id);
    if (existing != NULL && (size_t)existing->payload_len == len) {
        if (len > 0U) {
            memcpy(store->heap_data + existing->data_offset + NANOVEC_META_REC_HDR_SZ, payload, len);
        }
        nanovec_meta_unlock(store);
        return NANOVEC_SUCCESS;
    }

    size_t rec_sz = (size_t)NANOVEC_META_REC_HDR_SZ + len;
    st = nanovec_meta_ensure_heap_space(store, rec_sz);
    if (st != NANOVEC_SUCCESS) {
        nanovec_meta_unlock(store);
        return st;
    }

    uint64_t off = (uint64_t)store->data_used;
    uint8_t *dst = store->heap_data + off;
    nanovec_write_le64(dst + 0x00, (uint64_t)id);
    nanovec_write_le32(dst + 0x08, (uint32_t)len);
    if (len > 0U) {
        memcpy(dst + NANOVEC_META_REC_HDR_SZ, payload, len);
    }

    bool     replaced = false;
    uint32_t old_len  = 0U;
    st = nanovec_meta_upsert_slot(store, id, off, (uint32_t)len, &replaced, &old_len);
    if (st != NANOVEC_SUCCESS) {
        nanovec_meta_unlock(store);
        return st;
    }

    store->data_used += rec_sz;
    if (replaced) {
        store->dead_bytes += (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)old_len;
        if (store->data_used >= NANOVEC_META_AUTO_COMPACT_SZ &&
            store->dead_bytes * 2U > store->data_used) {
            (void)nanovec_meta_compact_locked(store);
        }
    }

    nanovec_meta_unlock(store);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_delete(
    nanovec_metadata_store_t *store,
    nanovec_vid_t             id
) {
    nanovec_status_t st = nanovec_meta_wrlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    if (NANOVEC_UNLIKELY(store->readonly)) {
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "cannot delete from read-only metadata store"
        );
    }

    nanovec_meta_slot_t *s = nanovec_meta_find_slot(store, id);
    if (s == NULL) {
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_NOT_FOUND;
    }

    s->state = NANOVEC_META_SLOT_TOMBSTONE;
    store->live_count--;
    store->tombstone_count++;
    store->dead_bytes += (size_t)NANOVEC_META_REC_HDR_SZ + (size_t)s->payload_len;

    if (store->data_used >= NANOVEC_META_AUTO_COMPACT_SZ &&
        store->dead_bytes * 2U > store->data_used) {
        (void)nanovec_meta_compact_locked(store);
    }

    nanovec_meta_unlock(store);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_get(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id,
    void                           *out_buf,
    size_t                          buf_cap,
    size_t                         *out_len
) {
    if (out_len != NULL) {
        *out_len = 0U;
    }
    if (NANOVEC_UNLIKELY(out_buf == NULL && buf_cap > 0U)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "out_buf NULL with non-zero cap");
    }

    nanovec_status_t st = nanovec_meta_rdlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    const nanovec_meta_slot_t *s = nanovec_meta_find_slot(store, id);
    if (s == NULL) {
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_NOT_FOUND;
    }

    size_t len = (size_t)s->payload_len;
    if (out_len != NULL) {
        *out_len = len;
    }

    if (len == 0U) {
        nanovec_meta_unlock(store);
        return NANOVEC_SUCCESS;
    }

    /* Size-query convention: out_buf == NULL, buf_cap == 0, out_len != NULL. */
    if (out_buf == NULL && buf_cap == 0U && out_len != NULL) {
        nanovec_meta_unlock(store);
        return NANOVEC_SUCCESS;
    }

    if (NANOVEC_UNLIKELY(out_buf == NULL || buf_cap < len)) {
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "metadata output buffer too small (%zu < %zu)",
            buf_cap,
            len
        );
    }

    const uint8_t *base = nanovec_meta_base_const(store);
    memcpy(out_buf, base + s->data_offset + NANOVEC_META_REC_HDR_SZ, len);

    nanovec_meta_unlock(store);
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_get_view(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id,
    const void                    **out_ptr,
    size_t                         *out_len
) {
    if (out_ptr != NULL) *out_ptr = NULL;
    if (out_len != NULL) *out_len = 0U;

    nanovec_status_t st = nanovec_meta_rdlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    const nanovec_meta_slot_t *s = nanovec_meta_find_slot(store, id);
    if (s == NULL) {
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_NOT_FOUND;
    }

    const uint8_t *base = nanovec_meta_base_const(store);
    if (out_ptr != NULL) {
        *out_ptr = (s->payload_len > 0U)
            ? (const void *)(base + s->data_offset + NANOVEC_META_REC_HDR_SZ)
            : NULL;
    }
    if (out_len != NULL) {
        *out_len = (size_t)s->payload_len;
    }

    nanovec_meta_unlock(store);
    return NANOVEC_SUCCESS;
}

bool nanovec_metadata_contains(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id
) {
    if (nanovec_meta_rdlock(store) != NANOVEC_SUCCESS) {
        return false;
    }
    bool found = (nanovec_meta_find_slot(store, id) != NULL);
    nanovec_meta_unlock(store);
    return found;
}

size_t nanovec_metadata_count(const nanovec_metadata_store_t *store) {
    if (nanovec_meta_rdlock(store) != NANOVEC_SUCCESS) {
        return 0U;
    }
    size_t cnt = store->live_count;
    nanovec_meta_unlock(store);
    return cnt;
}

nanovec_status_t nanovec_metadata_compact(nanovec_metadata_store_t *store) {
    nanovec_status_t st = nanovec_meta_wrlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    if (NANOVEC_UNLIKELY(store->readonly)) {
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "cannot compact read-only metadata store"
        );
    }
    st = nanovec_meta_compact_locked(store);
    nanovec_meta_unlock(store);
    return st;
}

nanovec_status_t nanovec_metadata_iterate(
    const nanovec_metadata_store_t *store,
    nanovec_metadata_iter_fn_t      cb,
    void                           *ctx
) {
    if (NANOVEC_UNLIKELY(cb == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_INVALID_ARG, "callback is NULL");
    }

    nanovec_status_t st = nanovec_meta_rdlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    const uint8_t *base = nanovec_meta_base_const(store);
    for (size_t i = 0U; i < store->slot_cap; ++i) {
        const nanovec_meta_slot_t *s = &store->slots[i];
        if (s->state != NANOVEC_META_SLOT_OCCUPIED) {
            continue;
        }
        const void *p = (s->payload_len > 0U)
            ? (const void *)(base + s->data_offset + NANOVEC_META_REC_HDR_SZ)
            : NULL;
        st = cb(s->id, p, (size_t)s->payload_len, ctx);
        if (st != NANOVEC_SUCCESS) {
            break;
        }
    }

    nanovec_meta_unlock(store);
    return st;
}

size_t nanovec_metadata_serialized_size(const nanovec_metadata_store_t *store) {
    if (nanovec_meta_rdlock(store) != NANOVEC_SUCCESS) {
        return 0U;
    }
    size_t sz = (size_t)NANOVEC_META_SECTION_HDR_SZ + nanovec_meta_live_bytes_locked(store);
    nanovec_meta_unlock(store);
    return sz;
}

nanovec_status_t nanovec_metadata_serialize(
    const nanovec_metadata_store_t *store,
    void                           *out_buf,
    size_t                          buf_cap,
    size_t                         *out_written
) {
    if (out_written != NULL) {
        *out_written = 0U;
    }
    nanovec_status_t st = nanovec_meta_rdlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }
    st = nanovec_meta_serialize_locked(store, out_buf, buf_cap, out_written);
    nanovec_meta_unlock(store);
    return st;
}

nanovec_status_t nanovec_metadata_deserialize(
    nanovec_metadata_store_t *store,
    const void               *buf,
    size_t                    buf_len
) {
    if (NANOVEC_UNLIKELY(store == NULL || buf == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid metadata_deserialize args"
        );
    }

    nanovec_meta_section_header_t hdr;
    nanovec_status_t st = nanovec_meta_parse_section((const uint8_t *)buf, buf_len, &hdr);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    uint64_t rec_count  = nanovec_le64toh(hdr.record_count);
    uint64_t data_bytes = nanovec_le64toh(hdr.data_bytes);

    size_t slot_cap = NANOVEC_META_MIN_SLOTS;
    while ((rec_count + 1ULL) * 10ULL >= (uint64_t)slot_cap * 7ULL && slot_cap <= SIZE_MAX / 2U) {
        slot_cap *= 2U;
    }

    size_t heap_cap = (data_bytes > (uint64_t)NANOVEC_META_DEFAULT_BYTES)
        ? (size_t)data_bytes
        : NANOVEC_META_DEFAULT_BYTES;

    st = nanovec_meta_init_common(store, slot_cap, heap_cap, false);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    if (data_bytes > 0ULL) {
        memcpy(
            store->heap_data,
            (const uint8_t *)buf + NANOVEC_META_SECTION_HDR_SZ,
            (size_t)data_bytes
        );
        store->data_used = (size_t)data_bytes;
    }

    st = nanovec_meta_index_records(store, rec_count);
    if (st != NANOVEC_SUCCESS) {
        nanovec_metadata_destroy(store);
        return st;
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_attach_mmap(
    nanovec_metadata_store_t *store,
    const nanovec_mmap_t     *map
) {
    if (NANOVEC_UNLIKELY(store == NULL || map == NULL || map->base == NULL)) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid metadata_attach_mmap args"
        );
    }

    const nanovec_file_header_t *fhdr = nanovec_mmap_header(map);
    if (NANOVEC_UNLIKELY(fhdr == NULL)) {
        return (nanovec_status_t)nanovec_set_error(NANOVEC_ERR_CORRUPT, "missing .nvec header");
    }

    uint64_t meta_off = nanovec_le64toh(fhdr->metadata_offset);
    if (NANOVEC_UNLIKELY(
            meta_off < NANOVEC_HEADER_SIZE || meta_off >= (uint64_t)map->mapped_len
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_NOT_FOUND,
            "metadata section absent or out of range"
        );
    }

    const uint8_t *sec_ptr = map->base + (size_t)meta_off;
    size_t         avail   = map->mapped_len - (size_t)meta_off;

    nanovec_meta_section_header_t hdr;
    nanovec_status_t st = nanovec_meta_parse_section(sec_ptr, avail, &hdr);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    uint64_t rec_count  = nanovec_le64toh(hdr.record_count);
    uint64_t data_bytes = nanovec_le64toh(hdr.data_bytes);

    size_t slot_cap = NANOVEC_META_MIN_SLOTS;
    while ((rec_count + 1ULL) * 10ULL >= (uint64_t)slot_cap * 7ULL && slot_cap <= SIZE_MAX / 2U) {
        slot_cap *= 2U;
    }

    st = nanovec_meta_init_common(store, slot_cap, 0U, true);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    store->mmap_data = sec_ptr + NANOVEC_META_SECTION_HDR_SZ;
    store->data_used = (size_t)data_bytes;
    store->data_cap  = (size_t)data_bytes;

    st = nanovec_meta_index_records(store, rec_count);
    if (st != NANOVEC_SUCCESS) {
        nanovec_metadata_destroy(store);
        return st;
    }
    return NANOVEC_SUCCESS;
}

nanovec_status_t nanovec_metadata_persist_nvec(
    const nanovec_metadata_store_t *store,
    nanovec_growable_map_t         *gmap,
    uint64_t                        metadata_offset,
    size_t                         *out_section_bytes
) {
    if (out_section_bytes != NULL) {
        *out_section_bytes = 0U;
    }
    if (NANOVEC_UNLIKELY(
            gmap == NULL || !gmap->lock_init || metadata_offset < (uint64_t)NANOVEC_HEADER_SIZE
        )) {
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "invalid metadata_persist_nvec args"
        );
    }

    nanovec_status_t st = nanovec_meta_rdlock(store);
    if (st != NANOVEC_SUCCESS) {
        return st;
    }

    size_t sec_sz = (size_t)NANOVEC_META_SECTION_HDR_SZ + nanovec_meta_live_bytes_locked(store);
    if (NANOVEC_UNLIKELY(metadata_offset > (uint64_t)SIZE_MAX - (uint64_t)sec_sz)) {
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_INVALID_ARG;
    }
    size_t req_end = (size_t)metadata_offset + sec_sz;

    st = nanovec_growth_ensure_capacity(gmap, req_end, NULL);
    if (st != NANOVEC_SUCCESS) {
        nanovec_meta_unlock(store);
        return st;
    }

    int rc = pthread_rwlock_wrlock(&gmap->rwlock);
    if (NANOVEC_UNLIKELY(rc != 0)) {
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_errno(NANOVEC_ERR_INTERNAL, rc, "rwlock wrlock failed");
    }

    if (NANOVEC_UNLIKELY(gmap->map.readonly)) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        nanovec_meta_unlock(store);
        return (nanovec_status_t)nanovec_set_error(
            NANOVEC_ERR_INVALID_ARG,
            "cannot persist metadata to read-only mapping"
        );
    }

    if (NANOVEC_UNLIKELY(gmap->map.base == NULL || req_end > gmap->map.mapped_len)) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_CORRUPT;
    }

    nanovec_file_header_t *hdr = nanovec_mmap_header(&gmap->map);
    if (NANOVEC_UNLIKELY(hdr == NULL)) {
        (void)pthread_rwlock_unlock(&gmap->rwlock);
        nanovec_meta_unlock(store);
        return NANOVEC_ERR_CORRUPT;
    }

    size_t written = 0U;
    st = nanovec_meta_serialize_locked(
        store,
        gmap->map.base + (size_t)metadata_offset,
        gmap->map.mapped_len - (size_t)metadata_offset,
        &written
    );
    if (st == NANOVEC_SUCCESS) {
        hdr->metadata_offset = nanovec_htole64(metadata_offset);
        uint16_t flags = nanovec_le16toh(hdr->flags);
        flags |= (uint16_t)NANOVEC_HDR_FLAG_HAS_METADATA;
        hdr->flags = nanovec_htole16(flags);
        nanovec_header_seal(hdr);
        gmap->ptrs.metadata = (void *)(gmap->map.base + (size_t)metadata_offset);
    }

    (void)pthread_rwlock_unlock(&gmap->rwlock);
    nanovec_meta_unlock(store);

    if (st == NANOVEC_SUCCESS) {
        st = nanovec_growth_refresh_ptrs(gmap);
        if (st == NANOVEC_SUCCESS && out_section_bytes != NULL) {
            *out_section_bytes = written;
        }
    }
    return st;
}
