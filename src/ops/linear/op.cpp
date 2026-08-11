#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/linear_iluvatar.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: out/in/weight must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    const std::byte *bias_ptr = (bias != nullptr) ? bias->data() : nullptr;

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           M, N, K, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           M, N, K, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              bias_ptr, M, N, K, out->dtype());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::linear(out->data(), in->data(), weight->data(),
                                bias_ptr, M, N, K, out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
