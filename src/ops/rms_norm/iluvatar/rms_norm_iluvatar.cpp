#include "rms_norm_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void rms_norm(std::byte *, const std::byte *, const std::byte *,
              size_t, size_t, float, llaisysDataType_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
