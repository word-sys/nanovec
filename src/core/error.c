/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#include "core/error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static _Thread_local char g_err_buf[NANOVEC_ERR_BUF_SIZE];
static _Thread_local int  g_err_code = NANOVEC_SUCCESS;

static const char *nanovec_strerror_ts(int errnum, char *buf, size_t len) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    const char *msg = strerror_r(errnum, buf, len);
    return msg ? msg : "unknown error";
#else
    if (strerror_r(errnum, buf, len) == 0) {
        return buf;
    }
    (void)snprintf(buf, len, "errno %d", errnum);
    return buf;
#endif
}

static size_t nanovec_vformat_err(int code, const char *fmt, va_list ap) {
    if (!fmt || fmt[0] == '\0') {
        const char *fallback = nanovec_status_string((nanovec_status_t)code);
        int n = snprintf(g_err_buf, sizeof(g_err_buf), "%s", fallback);
        if (n < 0) {
            g_err_buf[0] = '\0';
            return 0;
        }
        return (size_t)n >= sizeof(g_err_buf) ? sizeof(g_err_buf) - 1U : (size_t)n;
    }

    int n = vsnprintf(g_err_buf, sizeof(g_err_buf), fmt, ap);
    if (n < 0) {
        g_err_buf[0] = '\0';
        return 0;
    }
    return (size_t)n >= sizeof(g_err_buf) ? sizeof(g_err_buf) - 1U : (size_t)n;
}

int nanovec_set_error(int code, const char *fmt, ...) {
    va_list ap;
    g_err_code = code;
    va_start(ap, fmt);
    (void)nanovec_vformat_err(code, fmt, ap);
    va_end(ap);
    return code;
}

int nanovec_set_errno(int code, int errnum, const char *fmt, ...) {
    va_list ap;
    g_err_code = code;
    va_start(ap, fmt);
    size_t used = nanovec_vformat_err(code, fmt, ap);
    va_end(ap);

    if (errnum != 0 && used + 2U < sizeof(g_err_buf)) {
        char sys_buf[128];
        const char *sys_msg = nanovec_strerror_ts(errnum, sys_buf, sizeof(sys_buf));
        (void)snprintf(
            g_err_buf + used,
            sizeof(g_err_buf) - used,
            ": %s",
            sys_msg
        );
    }
    return code;
}

int nanovec_get_last_error_code(void) {
    return g_err_code;
}

NANOVEC_API const char *nanovec_get_last_error(void) {
    return g_err_buf;
}

NANOVEC_API void nanovec_clear_error(void) {
    g_err_code = NANOVEC_SUCCESS;
    g_err_buf[0] = '\0';
}

NANOVEC_API const char *nanovec_status_string(nanovec_status_t status) {
    switch (status) {
        case NANOVEC_SUCCESS:            return "success";
        case NANOVEC_ERR_INVALID_ARG:    return "invalid argument";
        case NANOVEC_ERR_OOM:            return "out of memory";
        case NANOVEC_ERR_IO:             return "I/O error";
        case NANOVEC_ERR_CORRUPT:        return "corrupt data";
        case NANOVEC_ERR_NOT_FOUND:      return "not found";
        case NANOVEC_ERR_ALREADY_EXISTS: return "already exists";
        case NANOVEC_ERR_FULL:           return "capacity full";
        case NANOVEC_ERR_DIM_MISMATCH:   return "dimension mismatch";
        case NANOVEC_ERR_UNSUPPORTED:    return "unsupported operation";
        case NANOVEC_ERR_MMAP:           return "memory map error";
        case NANOVEC_ERR_WAL:            return "write-ahead log error";
        case NANOVEC_ERR_MODEL:          return "model error";
        case NANOVEC_ERR_TOKENIZER:      return "tokenizer error";
        case NANOVEC_ERR_INTERNAL:       return "internal error";
        case NANOVEC_ERR_BUSY:           return "resource busy";
        case NANOVEC_ERR_VERSION:        return "incompatible version";
    }
    return "unknown error";
}
