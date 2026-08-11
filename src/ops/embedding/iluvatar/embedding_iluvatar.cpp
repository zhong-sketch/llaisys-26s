#include "embedding_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void embedding(std::byte *, const std::byte *, const std::byte *,
               size_t, size_t, size_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
