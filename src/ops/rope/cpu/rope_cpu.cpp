#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t d_half = d / 2;
    for (size_t s = 0; s < seqlen; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            size_t base = s * nhead * d + h * d;
            for (size_t j = 0; j < d_half; j++) {
                float phi     = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                float a = llaisys::utils::cast<float>(in[base + j]);
                float b = llaisys::utils::cast<float>(in[base + j + d_half]);
                out[base + j]          = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[base + j + d_half] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          size_t seqlen, size_t nhead, size_t d, float theta,
          llaisysDataType_t dtype) {
    const int64_t *ids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in), ids,
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in), ids,
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in), ids,
                     seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
