#include "rms_norm_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <math.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight,
                              size_t M, size_t d, float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum_sq = 0.0f;
        for (size_t k = 0; k < d; k++) {
            const float v = loadAsFloat(in[row * d + k]);
            sum_sq += v * v;
        }
        const float scale = rsqrtf(sum_sq / static_cast<float>(d) + eps);
        for (size_t k = 0; k < d; k++) {
            const float v = loadAsFloat(in[row * d + k]);
            const float w = loadAsFloat(weight[k]);
            out[row * d + k] = storeFromFloat<T>(v * scale * w);
        }
    }
}

template <typename T>
void rmsNormTyped(std::byte *out, const std::byte *in,
                  const std::byte *weight, size_t M, size_t d, float eps) {
    rmsNormKernel<<<blocksFor(M), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        M, d, eps);
    checkKernel("rmsNormKernel");
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t d, float eps, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rmsNormTyped<float>(out, in, weight, M, d, eps);
    case LLAISYS_DTYPE_F16:
        return rmsNormTyped<llaisys::fp16_t>(out, in, weight, M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rmsNormTyped<llaisys::bf16_t>(out, in, weight, M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
