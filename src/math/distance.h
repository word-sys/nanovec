#ifndef NANOVEC_MATH_DISTANCE_H
#define NANOVEC_MATH_DISTANCE_H

#include "core/types.h"

/* All nanovec_dist_* kernels follow smaller-is-closer ordering. */
float nanovec_dist_l2_scalar(const float *a, const float *b, uint32_t dim);
float nanovec_dist_dot_scalar(const float *a, const float *b, uint32_t dim);
float nanovec_dist_cosine_scalar(const float *a, const float *b, uint32_t dim);

#endif /* NANOVEC_MATH_DISTANCE_H */
