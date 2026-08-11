#include "swiglu_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void swiglu(std::byte *, const std::byte *, const std::byte *,
            llaisysDataType_t, size_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
