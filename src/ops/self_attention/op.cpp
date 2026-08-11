#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/self_attention_iluvatar.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
           k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t qlen  = q->shape()[0];
    size_t nh    = q->shape()[1];
    size_t d     = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh  = k->shape()[1];
    size_t dv    = v->shape()[2];

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
