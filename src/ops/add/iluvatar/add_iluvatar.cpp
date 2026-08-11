#include "add_iluvatar.hpp"

#include "../../iluvatar/unsupported.hpp"

namespace llaisys::ops::iluvatar {
void add(std::byte *, const std::byte *, const std::byte *,
         llaisysDataType_t, size_t) {
    throwUnsupported();
}
} // namespace llaisys::ops::iluvatar
