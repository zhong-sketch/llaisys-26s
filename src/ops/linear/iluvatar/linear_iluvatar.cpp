#include "linear_iluvatar.hpp"

#include "../../iluvatar/fallback.hpp"
#include "../../linear/cpu/linear_cpu.hpp"

namespace llaisys::ops::iluvatar {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype) {
    size_t elem_size = dtypeSize(dtype);
    auto host_in = copyFromDevice(in, M * K * elem_size);
    auto host_weight = copyFromDevice(weight, N * K * elem_size);
    auto host_bias = bias != nullptr
                         ? copyFromDevice(bias, N * elem_size)
                         : std::vector<std::byte>{};
    std::vector<std::byte> host_out(M * N * elem_size);

    cpu::linear(host_out.data(), host_in.data(), host_weight.data(),
                bias != nullptr ? host_bias.data() : nullptr,
                M, N, K, dtype);
    copyToDevice(out, host_out.data(), host_out.size());
}
} // namespace llaisys::ops::iluvatar
