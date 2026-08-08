#include "embedding_cpu.hpp"

#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    size_t row_bytes = embed_dim * elem_size;

    for (size_t i = 0; i < n; i++) {
        int64_t row = idx_ptr[i];
        std::memcpy(
            out    + i   * row_bytes,
            weight + row * row_bytes,
            row_bytes
        );
    }
}
} // namespace llaisys::ops::cpu
