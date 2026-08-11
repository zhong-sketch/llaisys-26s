#include "rms_norm_iluvatar.hpp"

#include "../../iluvatar/fallback.hpp"
#include "../../rms_norm/cpu/rms_norm_cpu.hpp"

namespace llaisys::ops::iluvatar {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t d, float eps, llaisysDataType_t dtype) {
    size_t elem_size = dtypeSize(dtype);
    auto host_in = copyFromDevice(in, M * d * elem_size);
    auto host_weight = copyFromDevice(weight, d * elem_size);
    std::vector<std::byte> host_out(M * d * elem_size);

    cpu::rms_norm(host_out.data(), host_in.data(), host_weight.data(),
                  M, d, eps, dtype);
    copyToDevice(out, host_out.data(), host_out.size());
}
} // namespace llaisys::ops::iluvatar
