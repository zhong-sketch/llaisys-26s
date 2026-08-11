#include "rope_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <cstdint>
#include <math.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids,
                           size_t seqlen, size_t nhead, size_t d,
                           float theta) {
    const size_t d_half = d / 2;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = seqlen * nhead * d_half;
    if (idx < total) {
        const size_t j = idx % d_half;
        const size_t h = (idx / d_half) % nhead;
        const size_t s = idx / (d_half * nhead);
        const size_t base = s * nhead * d + h * d;
        const float pos = static_cast<float>(pos_ids[s]);
        const float phi = pos / powf(theta, 2.0f * static_cast<float>(j) /
                                             static_cast<float>(d));
        float sin_phi = 0.0f;
        float cos_phi = 0.0f;
        sincosf(phi, &sin_phi, &cos_phi);
        const float a = loadAsFloat(in[base + j]);
        const float b = loadAsFloat(in[base + j + d_half]);
        out[base + j] = storeFromFloat<T>(a * cos_phi - b * sin_phi);
        out[base + j + d_half] = storeFromFloat<T>(b * cos_phi + a * sin_phi);
    }
}

template <typename T>
void ropeTyped(std::byte *out, const std::byte *in,
               const std::byte *pos_ids, size_t seqlen, size_t nhead,
               size_t d, float theta) {
    const size_t total = seqlen * nhead * (d / 2);
    ropeKernel<<<blocksFor(total), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        seqlen, nhead, d, theta);
    checkKernel("ropeKernel");
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          size_t seqlen, size_t nhead, size_t d, float theta,
          llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return ropeTyped<float>(out, in, pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return ropeTyped<llaisys::fp16_t>(out, in, pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return ropeTyped<llaisys::bf16_t>(out, in, pos_ids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
