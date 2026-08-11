#pragma once

#include <cstddef>

namespace llaisys::ops::iluvatar {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size);
}
