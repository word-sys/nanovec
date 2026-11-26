/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_STORAGE_CHECKSUM_H
#define NANOVEC_STORAGE_CHECKSUM_H

#include "core/types.h"
#include "storage/format.h"

uint32_t nanovec_crc32c_scalar(uint32_t crc, const void *data, size_t len);
uint32_t nanovec_crc32c_update(uint32_t crc, const void *data, size_t len);
uint32_t nanovec_crc32c(const void *data, size_t len);

uint64_t nanovec_xxh64(const void *data, size_t len, uint64_t seed);

uint32_t         nanovec_header_compute_crc32(const nanovec_file_header_t *hdr);
void             nanovec_header_seal(nanovec_file_header_t *hdr);
nanovec_status_t nanovec_header_verify(const nanovec_file_header_t *hdr);

#endif /* NANOVEC_STORAGE_CHECKSUM_H */
