#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/embedding_iluvatar.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "embedding: index must be Int64");

    size_t n         = index->numel();       // 行数 N
    size_t embed_dim = weight->shape()[1];   // 每行的元素数
    size_t elem_size = weight->elementSize(); // 每个元素的字节数

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              n, embed_dim, elem_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              n, embed_dim, elem_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                 n, embed_dim, elem_size);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::embedding(out->data(), index->data(), weight->data(),
                                   n, embed_dim, elem_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
