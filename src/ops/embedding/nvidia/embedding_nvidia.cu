#include "embedding_nvidia.cuh"

#include "../../nvidia/common.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {

__global__ void embeddingKernel(std::byte *out, const int64_t *index,
                                const std::byte *weight, size_t row_bytes,
                                size_t total_bytes) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_bytes) {
        const size_t token = i / row_bytes;
        const size_t col_byte = i % row_bytes;
        const int64_t row = index[token];
        out[i] = weight[static_cast<size_t>(row) * row_bytes + col_byte];
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size) {
    const size_t row_bytes = embed_dim * elem_size;
    const size_t total_bytes = n * row_bytes;
    embeddingKernel<<<blocksFor(total_bytes), THREADS_PER_BLOCK>>>(
        out, reinterpret_cast<const int64_t *>(index), weight,
        row_bytes, total_bytes);
    checkKernel("embeddingKernel");
}

} // namespace llaisys::ops::nvidia
