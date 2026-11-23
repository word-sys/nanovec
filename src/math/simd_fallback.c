#include "math/distance.h"

float nanovec_dist_l2_scalar(const float *a, const float *b, uint32_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    uint32_t i = 0;
    uint32_t limit = dim & ~3U;
    for (; i < limit; i += 4U) {
        float d0 = a[i]     - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    float tail = 0.0f;
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        tail += d * d;
    }

    return ((sum0 + sum1) + (sum2 + sum3)) + tail;
}
