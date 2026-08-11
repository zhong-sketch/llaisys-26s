#include "argmax_iluvatar.hpp"

#include "../../argmax/cpu/argmax_cpu.hpp"
#include "../../iluvatar/fallback.hpp"

namespace llaisys::ops::iluvatar {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    size_t val_bytes = dtypeSize(dtype);
    auto host_vals = copyFromDevice(vals, numel * val_bytes);
    std::vector<std::byte> host_idx(sizeof(int64_t));
    std::vector<std::byte> host_val(val_bytes);

    cpu::argmax(host_idx.data(), host_val.data(), host_vals.data(),
                dtype, numel);
    copyToDevice(max_idx, host_idx.data(), host_idx.size());
    copyToDevice(max_val, host_val.data(), host_val.size());
}
} // namespace llaisys::ops::iluvatar
