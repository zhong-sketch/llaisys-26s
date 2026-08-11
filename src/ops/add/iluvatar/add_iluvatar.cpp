#include "add_iluvatar.hpp"

#include "../../add/cpu/add_cpu.hpp"
#include "../../iluvatar/fallback.hpp"

namespace llaisys::ops::iluvatar {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel) {
    size_t bytes = numel * dtypeSize(dtype);
    auto host_a = copyFromDevice(a, bytes);
    auto host_b = copyFromDevice(b, bytes);
    std::vector<std::byte> host_c(bytes);

    cpu::add(host_c.data(), host_a.data(), host_b.data(), dtype, numel);
    copyToDevice(c, host_c.data(), bytes);
}
} // namespace llaisys::ops::iluvatar
