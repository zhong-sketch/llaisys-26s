#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_ARGUMENT(out->dtype() == in->dtype(),
                   "rearrange: dtype mismatch");
    CHECK_ARGUMENT(out->numel() == in->numel(),
                   "rearrange: element count mismatch");
    ASSERT(out->isContiguous() && in->isContiguous(),
           "rearrange: tensors must be contiguous");

    core::context().setDevice(out->deviceType(), out->deviceId());
    const size_t bytes = out->numel() * out->elementSize();
    const auto kind = out->deviceType() == LLAISYS_DEVICE_CPU
                          ? LLAISYS_MEMCPY_H2H
                          : LLAISYS_MEMCPY_D2D;
    core::context().runtime().api()->memcpy_sync(
        out->data(), in->data(), bytes, kind);
}
} // namespace llaisys::ops
