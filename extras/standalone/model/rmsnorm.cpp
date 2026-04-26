#include "rmsnorm.h"
#include "../math/vec_ops.h"
#include <cmath>

namespace turbocpp {

void rmsnorm_row(const float* x, const float* weight, float* y,
                 size_t dim, float eps) {
    // Use the SIMD-accelerated mean_sq from vec_ops. Then scale into y
    // via element-wise multiply — we write a single pass over the row.
    const float ms = mean_sq(x, dim);
    const float scale = 1.0f / std::sqrt(ms + eps);
    // y[i] = x[i] * weight[i] * scale.  Do this with two fused ops:
    //   y = x * weight  (vec_mul is SIMD)
    //   y *= scale
    vec_mul(y, x, weight, dim);
    vec_scale(y, scale, dim);
}

void rmsnorm(const float* x, const float* weight, float* y,
             size_t rows, size_t dim, float eps) {
    for (size_t r = 0; r < rows; ++r) {
        rmsnorm_row(x + r * dim, weight, y + r * dim, dim, eps);
    }
}

} // namespace turbocpp
