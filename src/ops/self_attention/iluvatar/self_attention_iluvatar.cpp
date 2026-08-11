#include "self_attention_iluvatar.hpp"

#include "../../iluvatar/fallback.hpp"
#include "../../self_attention/cpu/self_attention_cpu.hpp"

namespace llaisys::ops::iluvatar {
void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k,
                    const std::byte *v, size_t qlen, size_t nh,
                    size_t kvlen, size_t nkvh, size_t d, size_t dv,
                    float scale, llaisysDataType_t dtype) {
    size_t elem_size = dtypeSize(dtype);
    auto host_q = copyFromDevice(q, qlen * nh * d * elem_size);
    auto host_k = copyFromDevice(k, kvlen * nkvh * d * elem_size);
    auto host_v = copyFromDevice(v, kvlen * nkvh * dv * elem_size);
    std::vector<std::byte> host_attn_val(qlen * nh * dv * elem_size);

    cpu::self_attention(host_attn_val.data(), host_q.data(), host_k.data(),
                        host_v.data(), qlen, nh, kvlen, nkvh, d, dv,
                        scale, dtype);
    copyToDevice(attn_val, host_attn_val.data(), host_attn_val.size());
}
} // namespace llaisys::ops::iluvatar
