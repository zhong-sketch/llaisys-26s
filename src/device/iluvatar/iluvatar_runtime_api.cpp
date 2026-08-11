#include "../runtime_api.hpp"

#include <stdexcept>

namespace llaisys::device::iluvatar {

namespace runtime_api {
int getDeviceCount() {
    return 0;
}

void setDevice(int) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void deviceSynchronize() {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

llaisysStream_t createStream() {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void destroyStream(llaisysStream_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void streamSynchronize(llaisysStream_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void *mallocDevice(size_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void freeDevice(void *) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void *mallocHost(size_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void freeHost(void *) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void memcpySync(void *, const void *, size_t, llaisysMemcpyKind_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

void memcpyAsync(void *, const void *, size_t, llaisysMemcpyKind_t,
                 llaisysStream_t) {
    throw std::runtime_error(
        "Iluvatar runtime is scaffolded; configure the Iluvatar SDK first");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::iluvatar
