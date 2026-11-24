/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright (c) 2026 Barın Güzeldemirci */

#ifndef NANOVEC_CORE_MATH_FAST_H
#define NANOVEC_CORE_MATH_FAST_H

#include "core/types.h"

float nanovec_fast_rsqrt(float x);
float nanovec_fast_exp(float x);
float nanovec_fast_log(float x);
float nanovec_fast_tanh(float x);
float nanovec_fast_sigmoid(float x);

void nanovec_fast_rsqrt_vec(const float *in, float *out, size_t n);
void nanovec_fast_exp_vec(const float *in, float *out, size_t n);
void nanovec_fast_log_vec(const float *in, float *out, size_t n);

#endif /* NANOVEC_CORE_MATH_FAST_H */
