/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_METADATA_H
#define NANOVEC_STORAGE_METADATA_H

#include "core/types.h"
#include "storage/format.h"
#include "storage/growth.h"
#include "storage/mmap.h"

#include <pthread.h>

#define NANOVEC_META_MAGIC_0        0x4EU /* 'N' */
#define NANOVEC_META_MAGIC_1        0x56U /* 'V' */
#define NANOVEC_META_MAGIC_2        0x4DU /* 'M' */
#define NANOVEC_META_MAGIC_3        0x44U /* 'D' */
#define NANOVEC_META_VERSION        1U
#define NANOVEC_META_SECTION_HDR_SZ 32U
#define NANOVEC_META_REC_HDR_SZ     12U
#define NANOVEC_META_MAX_PAYLOAD    (64U * 1024U * 1024U)

typedef struct NANOVEC_PACKED nanovec_meta_section_header {
    uint8_t  magic[4];
    uint16_t version;
    uint16_t flags;
    uint64_t record_count;
    uint64_t data_bytes;
    uint32_t data_crc32;
    uint32_t header_crc32;
} nanovec_meta_section_header_t;

NANOVEC_STATIC_ASSERT(
    sizeof(nanovec_meta_section_header_t) == NANOVEC_META_SECTION_HDR_SZ,
    "metadata section header must be 32 bytes"
);

typedef struct nanovec_meta_slot {
    uint64_t id;
    uint64_t data_offset;
    uint32_t payload_len;
    uint8_t  state;
    uint8_t  _pad[3];
} nanovec_meta_slot_t;

typedef struct nanovec_metadata_store {
    nanovec_meta_slot_t *slots;
    size_t               slot_cap;
    size_t               live_count;
    size_t               tombstone_count;
    uint8_t             *heap_data;
    const uint8_t       *mmap_data;
    size_t               data_used;
    size_t               data_cap;
    size_t               dead_bytes;
    bool                 readonly;
    bool                 lock_init;
    bool                 initialized;
    pthread_rwlock_t     rwlock;
} nanovec_metadata_store_t;

typedef nanovec_status_t (*nanovec_metadata_iter_fn_t)(
    nanovec_vid_t id,
    const void   *payload,
    size_t        len,
    void         *ctx
);

nanovec_status_t nanovec_metadata_init(
    nanovec_metadata_store_t *store,
    size_t                    initial_bytes
);
void nanovec_metadata_destroy(nanovec_metadata_store_t *store);

nanovec_status_t nanovec_metadata_put(
    nanovec_metadata_store_t *store,
    nanovec_vid_t             id,
    const void               *payload,
    size_t                    len
);

nanovec_status_t nanovec_metadata_delete(
    nanovec_metadata_store_t *store,
    nanovec_vid_t             id
);

nanovec_status_t nanovec_metadata_get(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id,
    void                           *out_buf,
    size_t                          buf_cap,
    size_t                         *out_len
);

nanovec_status_t nanovec_metadata_get_view(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id,
    const void                    **out_ptr,
    size_t                         *out_len
);

bool nanovec_metadata_contains(
    const nanovec_metadata_store_t *store,
    nanovec_vid_t                   id
);

size_t nanovec_metadata_count(const nanovec_metadata_store_t *store);

nanovec_status_t nanovec_metadata_compact(nanovec_metadata_store_t *store);

nanovec_status_t nanovec_metadata_iterate(
    const nanovec_metadata_store_t *store,
    nanovec_metadata_iter_fn_t      cb,
    void                           *ctx
);

size_t nanovec_metadata_serialized_size(const nanovec_metadata_store_t *store);

nanovec_status_t nanovec_metadata_serialize(
    const nanovec_metadata_store_t *store,
    void                           *out_buf,
    size_t                          buf_cap,
    size_t                         *out_written
);

nanovec_status_t nanovec_metadata_deserialize(
    nanovec_metadata_store_t *store,
    const void               *buf,
    size_t                    buf_len
);

nanovec_status_t nanovec_metadata_attach_mmap(
    nanovec_metadata_store_t *store,
    const nanovec_mmap_t     *map
);

nanovec_status_t nanovec_metadata_persist_nvec(
    const nanovec_metadata_store_t *store,
    nanovec_growable_map_t         *gmap,
    uint64_t                        metadata_offset,
    size_t                         *out_section_bytes
);

#endif /* NANOVEC_STORAGE_METADATA_H */
