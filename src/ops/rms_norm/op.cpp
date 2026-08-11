#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rms_norm_iluvatar.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];  // 行数
    size_t d = in->shape()[1];  // 每行的元素数（归一化维度）

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             M, d, eps, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             M, d, eps, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                M, d, eps, out->dtype());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::rms_norm(out->data(), in->data(), weight->data(),
                                  M, d, eps, out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
