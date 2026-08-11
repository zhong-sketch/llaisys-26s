#include "swiglu_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <math.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up,
                             size_t numel) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        const float g = loadAsFloat(gate[i]);
        const float u = loadAsFloat(up[i]);
        const float sigmoid = 1.0f / (1.0f + expf(-g));
        out[i] = storeFromFloat<T>(u * g * sigmoid);
    }
}

template <typename T>
void swigluTyped(std::byte *out, const std::byte *gate, const std::byte *up,
                 size_t numel) {
    swigluKernel<<<blocksFor(numel), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    checkKernel("swigluKernel");
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swigluTyped<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swigluTyped<llaisys::fp16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swigluTyped<llaisys::bf16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
