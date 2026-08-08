#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val,
                     const T *q, const T *k, const T *v,
                     size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                     size_t d, size_t dv, float scale) {
    // 辅助缓冲区：每个 head 的注意力分数矩阵 [qlen × kvlen]
    std::vector<float> attn_scores(qlen * kvlen);

    for (size_t h = 0; h < nh; h++) {
        size_t h_kv = h * nkvh / nh;  // GQA：映射到对应的 kv-head

        // ---- Step 1: A = Q_h × K_hkv^T × scale + causal_mask ----
        for (size_t i = 0; i < qlen; i++) {
            for (size_t j = 0; j < kvlen; j++) {
                float dot = 0.0f;
                for (size_t dk = 0; dk < d; dk++) {
                    float qv = llaisys::utils::cast<float>(q[i * nh * d + h * d + dk]);
                    float kv = llaisys::utils::cast<float>(k[j * nkvh * d + h_kv * d + dk]);
                    dot += qv * kv;
                }
                // Causal mask: q 的第 i 行对应全局位置 kvlen-qlen+i，只能看 j <= 该位置
                bool masked = (j > kvlen - qlen + i);
                attn_scores[i * kvlen + j] = masked ? -std::numeric_limits<float>::infinity()
                                                     : dot * scale;
            }
        }

        // ---- Step 2: Numerically stable softmax over last dim ----
        for (size_t i = 0; i < qlen; i++) {
            // 找最大值（避免 exp 溢出）
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; j++) {
                if (attn_scores[i * kvlen + j] > max_val)
                    max_val = attn_scores[i * kvlen + j];
            }
            // exp 并求和
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[i * kvlen + j] = std::exp(attn_scores[i * kvlen + j] - max_val);
                sum += attn_scores[i * kvlen + j];
            }
            // 归一化
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[i * kvlen + j] /= sum;
            }
        }

        // ---- Step 3: out_h = attn_scores × V_hkv ----
        for (size_t i = 0; i < qlen; i++) {
            for (size_t dvi = 0; dvi < dv; dvi++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    acc += attn_scores[i * kvlen + j] *
                           llaisys::utils::cast<float>(v[j * nkvh * dv + h_kv * dv + dvi]);
                }
                attn_val[i * nh * dv + h * dv + dvi] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                    size_t d, size_t dv, float scale,
                    llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               qlen, nh, kvlen, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, nh, kvlen, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, nh, kvlen, nkvh, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
