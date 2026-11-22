#ifndef NANOVEC_CORE_ERROR_H
#define NANOVEC_CORE_ERROR_H

#include "core/types.h"

#define NANOVEC_ERR_BUF_SIZE 512U

#if defined(__GNUC__) || defined(__clang__)
#define NANOVEC_PRINTF_FMT(fmt_idx, arg_idx) \
    __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define NANOVEC_PRINTF_FMT(fmt_idx, arg_idx)
#endif

int nanovec_set_error(int code, const char *fmt, ...) NANOVEC_PRINTF_FMT(2, 3);
int nanovec_set_errno(int code, int errnum, const char *fmt, ...) NANOVEC_PRINTF_FMT(3, 4);
int nanovec_get_last_error_code(void);

#endif /* NANOVEC_CORE_ERROR_H */
