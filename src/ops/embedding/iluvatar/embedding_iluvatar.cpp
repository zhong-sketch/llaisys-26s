#include "embedding_iluvatar.hpp"

#include "../../embedding/cpu/embedding_cpu.hpp"
#include "../../iluvatar/fallback.hpp"

#include <algorithm>
#include <cstdint>

namespace llaisys::ops::iluvatar {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size) {
    size_t index_bytes = n * sizeof(int64_t);
    size_t row_bytes = embed_dim * elem_size;
    auto host_index = copyFromDevice(index, index_bytes);

    const auto *idx = reinterpret_cast<const int64_t *>(host_index.data());
    int64_t max_row = 0;
    for (size_t i = 0; i < n; i++) {
        max_row = std::max(max_row, idx[i]);
    }

    auto host_weight = copyFromDevice(
        weight, static_cast<size_t>(max_row + 1) * row_bytes);
    std::vector<std::byte> host_out(n * row_bytes);

    cpu::embedding(host_out.data(), host_index.data(), host_weight.data(),
                   n, embed_dim, elem_size);
    copyToDevice(out, host_out.data(), host_out.size());
}
} // namespace llaisys::ops::iluvatar
