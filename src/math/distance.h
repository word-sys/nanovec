#ifndef NANOVEC_MATH_DISTANCE_H
#define NANOVEC_MATH_DISTANCE_H

#include "core/types.h"

/* All nanovec_dist_* kernels follow smaller-is-closer ordering. */
float    nanovec_dist_l2_scalar(const float *a, const float *b, uint32_t dim);
float    nanovec_dist_dot_scalar(const float *a, const float *b, uint32_t dim);
float    nanovec_dist_cosine_scalar(const float *a, const float *b, uint32_t dim);
uint32_t nanovec_dist_hamming_scalar(const uint64_t *a, const uint64_t *b, uint32_t num_words);

#if defined(__x86_64__) || defined(__i386__)
float nanovec_dist_l2_avx2(const float *a, const float *b, uint32_t dim);
float nanovec_dist_dot_avx2(const float *a, const float *b, uint32_t dim);
#endif

#endif /* NANOVEC_MATH_DISTANCE_H */
