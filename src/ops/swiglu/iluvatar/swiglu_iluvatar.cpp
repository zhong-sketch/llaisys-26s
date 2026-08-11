#include "swiglu_iluvatar.hpp"

#include "../../iluvatar/fallback.hpp"
#include "../../swiglu/cpu/swiglu_cpu.hpp"

namespace llaisys::ops::iluvatar {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    size_t bytes = numel * dtypeSize(dtype);
    auto host_gate = copyFromDevice(gate, bytes);
    auto host_up = copyFromDevice(up, bytes);
    std::vector<std::byte> host_out(bytes);

    cpu::swiglu(host_out.data(), host_gate.data(), host_up.data(),
                dtype, numel);
    copyToDevice(out, host_out.data(), bytes);
}
} // namespace llaisys::ops::iluvatar
