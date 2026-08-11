#include "rope_iluvatar.hpp"

#include "../../iluvatar/fallback.hpp"
#include "../../rope/cpu/rope_cpu.hpp"

namespace llaisys::ops::iluvatar {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          size_t seqlen, size_t nhead, size_t d, float theta,
          llaisysDataType_t dtype) {
    size_t elem_size = dtypeSize(dtype);
    size_t data_bytes = seqlen * nhead * d * elem_size;
    auto host_in = copyFromDevice(in, data_bytes);
    auto host_pos_ids = copyFromDevice(pos_ids, seqlen * sizeof(int64_t));
    std::vector<std::byte> host_out(data_bytes);

    cpu::rope(host_out.data(), host_in.data(), host_pos_ids.data(),
              seqlen, nhead, d, theta, dtype);
    copyToDevice(out, host_out.data(), host_out.size());
}
} // namespace llaisys::ops::iluvatar
