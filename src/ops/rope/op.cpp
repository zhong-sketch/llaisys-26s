#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rope_iluvatar.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "rope: pos_ids must be Int64");

    size_t seqlen = in->shape()[0];
    size_t nhead  = in->shape()[1];
    size_t d      = in->shape()[2];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         seqlen, nhead, d, theta, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         seqlen, nhead, d, theta, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(),
                            seqlen, nhead, d, theta, out->dtype());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::rope(out->data(), in->data(), pos_ids->data(),
                              seqlen, nhead, d, theta, out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
