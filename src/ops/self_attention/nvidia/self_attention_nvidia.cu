#include "self_attention_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <float.h>
#include <math.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void selfAttentionKernel(T *attn_val, const T *q, const T *k,
                                    const T *v, size_t qlen, size_t nh,
                                    size_t kvlen, size_t nkvh, size_t d,
                                    size_t dv, float scale) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = qlen * nh * dv;
    if (idx >= total) {
        return;
    }

    const size_t dvi = idx % dv;
    const size_t h = (idx / dv) % nh;
    const size_t i = idx / (dv * nh);
    const size_t h_kv = h * nkvh / nh;
    const size_t causal_pos = kvlen - qlen + i;

    float max_score = -FLT_MAX;
    for (size_t j = 0; j < kvlen; j++) {
        float score = -INFINITY;
        if (j <= causal_pos) {
            float dot = 0.0f;
            for (size_t dk = 0; dk < d; dk++) {
                dot += loadAsFloat(q[i * nh * d + h * d + dk]) *
                       loadAsFloat(k[j * nkvh * d + h_kv * d + dk]);
            }
            score = dot * scale;
        }
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (size_t j = 0; j < kvlen; j++) {
        if (j <= causal_pos) {
            float dot = 0.0f;
            for (size_t dk = 0; dk < d; dk++) {
                dot += loadAsFloat(q[i * nh * d + h * d + dk]) *
                       loadAsFloat(k[j * nkvh * d + h_kv * d + dk]);
            }
            const float weight = expf(dot * scale - max_score);
            denom += weight;
            acc += weight * loadAsFloat(v[j * nkvh * dv + h_kv * dv + dvi]);
        }
    }

    attn_val[idx] = storeFromFloat<T>(acc / denom);
}

template <typename T>
void selfAttentionTyped(std::byte *attn_val, const std::byte *q,
                        const std::byte *k, const std::byte *v,
                        size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                        size_t d, size_t dv, float scale) {
    const size_t total = qlen * nh * dv;
    selfAttentionKernel<<<blocksFor(total), THREADS_PER_BLOCK>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        qlen, nh, kvlen, nkvh, d, dv, scale);
    checkKernel("selfAttentionKernel");
}

void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k,
                    const std::byte *v, size_t qlen, size_t nh,
                    size_t kvlen, size_t nkvh, size_t d, size_t dv,
                    float scale, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return selfAttentionTyped<float>(
            attn_val, q, k, v, qlen, nh, kvlen, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return selfAttentionTyped<llaisys::fp16_t>(
            attn_val, q, k, v, qlen, nh, kvlen, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return selfAttentionTyped<llaisys::bf16_t>(
            attn_val, q, k, v, qlen, nh, kvlen, nkvh, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
