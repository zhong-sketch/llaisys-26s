#pragma once

#include <stdexcept>

namespace llaisys::ops::iluvatar {
inline void throwUnsupported() {
    throw std::runtime_error(
        "Iluvatar op is scaffolded; configure the Iluvatar SDK first");
}
} // namespace llaisys::ops::iluvatar
