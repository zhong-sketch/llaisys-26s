#include "linear_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void linear(std::byte *, const std::byte *, const std::byte *,
            const std::byte *, size_t, size_t, size_t, llaisysDataType_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
