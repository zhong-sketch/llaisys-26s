#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; i++) {
        // 1. 计算该行的均方
        float sum_sq = 0.0f;
        for (size_t k = 0; k < d; k++) {
            float v = llaisys::utils::cast<float>(in[i * d + k]);
            sum_sq += v * v;
        }
        // 2. rsqrt(mean(x^2) + eps)
        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 3. 逐元素缩放并乘以权重
        for (size_t k = 0; k < d; k++) {
            float v = llaisys::utils::cast<float>(in[i * d + k]);
            float w = llaisys::utils::cast<float>(weight[k]);
            out[i * d + k] = llaisys::utils::cast<T>(v * scale * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t d, float eps, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), M, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
