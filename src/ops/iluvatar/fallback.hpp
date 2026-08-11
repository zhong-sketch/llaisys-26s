#pragma once

#include "llaisys.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaisys::ops::iluvatar {
inline void checkCuda(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(error));
    }
}

inline size_t dtypeSize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return 4;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return 2;
    case LLAISYS_DTYPE_I64:
        return 8;
    default:
        throw std::invalid_argument("Unsupported Iluvatar fallback dtype");
    }
}

inline std::vector<std::byte> copyFromDevice(const std::byte *device,
                                             size_t bytes) {
    std::vector<std::byte> host(bytes);
    if (bytes != 0) {
        checkCuda(cudaMemcpy(host.data(), device, bytes,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H");
    }
    return host;
}

inline void copyToDevice(std::byte *device, const std::byte *host,
                         size_t bytes) {
    if (bytes != 0) {
        checkCuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D");
    }
}
} // namespace llaisys::ops::iluvatar
