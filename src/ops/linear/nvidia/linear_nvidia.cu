#include "linear_nvidia.cuh"

#include "../../nvidia/common.cuh"

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void linearKernel(T *out, const T *in, const T *weight,
                             const T *bias, size_t M, size_t N, size_t K) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = M * N;
    if (idx < total) {
        const size_t i = idx / N;
        const size_t j = idx % N;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++) {
            acc += loadAsFloat(in[i * K + k]) *
                   loadAsFloat(weight[j * K + k]);
        }
        if (bias != nullptr) {
            acc += loadAsFloat(bias[j]);
        }
        out[idx] = storeFromFloat<T>(acc);
    }
}

template <typename T>
void linearTyped(std::byte *out, const std::byte *in, const std::byte *weight,
                 const std::byte *bias, size_t M, size_t N, size_t K) {
    const size_t total = M * N;
    linearKernel<<<blocksFor(total), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        reinterpret_cast<const T *>(bias),
        M, N, K);
    checkKernel("linearKernel");
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linearTyped<float>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_F16:
        return linearTyped<llaisys::fp16_t>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linearTyped<llaisys::bf16_t>(out, in, weight, bias, M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
