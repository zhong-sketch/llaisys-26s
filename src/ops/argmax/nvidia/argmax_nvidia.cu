#include "argmax_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <cstdint>
#include <float.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals,
                             size_t numel) {
    float best = -FLT_MAX;
    int64_t best_idx = 0;
    for (size_t i = 0; i < numel; i++) {
        const float v = loadAsFloat(vals[i]);
        if (v > best) {
            best = v;
            best_idx = static_cast<int64_t>(i);
        }
    }
    *max_idx = best_idx;
    *max_val = storeFromFloat<T>(best);
}

template <typename T>
void argmaxTyped(std::byte *max_idx, std::byte *max_val,
                 const std::byte *vals, size_t numel) {
    argmaxKernel<<<1, 1>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    checkKernel("argmaxKernel");
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmaxTyped<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmaxTyped<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmaxTyped<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
