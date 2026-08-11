#pragma once

#include "llaisys.h"

#include "../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {

constexpr int THREADS_PER_BLOCK = 256;

inline int blocksFor(size_t n) {
    return static_cast<int>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

inline void checkCuda(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(error));
    }
}

inline void checkKernel(const char *what) {
    checkCuda(cudaGetLastError(), what);
}

__device__ inline float f16BitsToFloat(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;

    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

__device__ inline uint16_t floatToF16Bits(float value) {
    const uint32_t x = __float_as_uint(value);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        const uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) |
        ((mant + 0x00001000u) >> 13));
}

__device__ inline float bf16BitsToFloat(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

__device__ inline uint16_t floatToBf16Bits(float value) {
    const uint32_t x = __float_as_uint(value);
    const uint32_t lsb = (x >> 16) & 1u;
    return static_cast<uint16_t>((x + 0x7fffu + lsb) >> 16);
}

template <typename T>
__device__ inline float loadAsFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float loadAsFloat<llaisys::fp16_t>(llaisys::fp16_t value) {
    return f16BitsToFloat(value._v);
}

template <>
__device__ inline float loadAsFloat<llaisys::bf16_t>(llaisys::bf16_t value) {
    return bf16BitsToFloat(value._v);
}

template <typename T>
__device__ inline T storeFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline llaisys::fp16_t storeFromFloat<llaisys::fp16_t>(float value) {
    llaisys::fp16_t out{};
    out._v = floatToF16Bits(value);
    return out;
}

template <>
__device__ inline llaisys::bf16_t storeFromFloat<llaisys::bf16_t>(float value) {
    llaisys::bf16_t out{};
    out._v = floatToBf16Bits(value);
    return out;
}

} // namespace llaisys::ops::nvidia
