#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::iluvatar {

namespace {
void checkCuda(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(error));
    }
}

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Unsupported Iluvatar memcpy kind");
    }
}
} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    checkCuda(error, "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    checkCuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)),
              "cudaStreamDestroy");
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
              "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkCuda(cudaFree(ptr), "cudaFree");
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkCuda(cudaFreeHost(ptr), "cudaFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)),
              "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    checkCuda(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind),
                              reinterpret_cast<cudaStream_t>(stream)),
              "cudaMemcpyAsync");
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
