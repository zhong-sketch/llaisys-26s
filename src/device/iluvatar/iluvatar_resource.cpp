#include "iluvatar_resource.hpp"

namespace llaisys::device::iluvatar {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_ILUVATAR, device_id) {}

Resource::~Resource() = default;

} // namespace llaisys::device::iluvatar
