#include "argmax_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void argmax(std::byte *, std::byte *, const std::byte *,
            llaisysDataType_t, size_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
