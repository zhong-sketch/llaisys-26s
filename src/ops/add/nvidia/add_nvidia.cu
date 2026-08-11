#include "add_nvidia.cuh"

#include "../../nvidia/common.cuh"

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        const float av = loadAsFloat(a[i]);
        const float bv = loadAsFloat(b[i]);
        c[i] = storeFromFloat<T>(av + bv);
    }
}

template <typename T>
void addTyped(std::byte *c, const std::byte *a, const std::byte *b,
              size_t numel) {
    addKernel<<<blocksFor(numel), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    checkKernel("addKernel");
}

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return addTyped<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return addTyped<llaisys::fp16_t>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return addTyped<llaisys::bf16_t>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
