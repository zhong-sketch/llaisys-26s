# 任务 2.5 — `void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta)`

## 任务本质

旋转位置编码（RoPE）是一种**将位置信息以旋转矩阵的形式编码进注意力向量**的方法。它不是加法式的位置编码（如 sin/cos positional embedding），而是通过对每个 head 的特征向量做 2D 旋转来实现，使得注意力的内积结果自然包含相对位置信息。

具体来说：将每个 head 的 `d` 维向量拆分成 `d/2` 对 `(a_j, b_j)`，对每对按角度 `φ_{i,j} = pos_i / θ^(2j/d)` 做二维旋转：

```
a'_{i,j} = a_{i,j} * cos(φ_{i,j}) - b_{i,j} * sin(φ_{i,j})
b'_{i,j} = b_{i,j} * cos(φ_{i,j}) + a_{i,j} * sin(φ_{i,j})
```

等价于：前半维度和后半维度分别参与旋转，但配对是"前 d/2 个" 配对 "后 d/2 个"。

## 目标

为输入张量 `in` 的每个向量（与 `pos_ids` 中的位置 id 对应）计算旋转位置编码：

- `out`：结果 q 或 k 张量，形状 `[seqlen, nhead, d]`
- `in`：原始 q 或 k 张量，形状 `[seqlen, nhead, d]`，连续
- `pos_ids`：每个 token 的位置 id，形状 `[seqlen,]`，dtype 为 int64
- `theta`：频率向量的基值（通常为 10000.0）

## 背景分析

### 测试规格

```python
testShapes = [
    ((2, 1, 4),       (0, 2)),    # seqlen=2, nhead=1, d=4, pos_ids=[0,1]
    ((512, 4, 4096),  (512, 1024)), # seqlen=512, nhead=4, d=4096, pos_ids=[512..1023]
]
# pos_ids = arrange(start, end) → 整数序列 [start, end)
```

### 参数维度

| 参数 | shape | dtype | 说明 |
|------|-------|-------|------|
| `out` | `[seqlen, nhead, d]` | f32/f16/bf16 | 输出张量 |
| `in` | `[seqlen, nhead, d]` | f32/f16/bf16 | 输入张量 |
| `pos_ids` | `[seqlen,]` | i64 | 每个 token 的全局位置 id |
| `theta` | float | — | RoPE 基频，通常 10000.0 |

### 关键算法（对应 torch_rope 实现）

```
d_half = d / 2
for s in 0..seqlen:
    pos = float(pos_ids[s])
    for h in 0..nhead:
        for j in 0..d_half:
            phi = pos / pow(theta, 2.0*j / d)
            cos_phi = cos(phi),  sin_phi = sin(phi)
            a = float(in[s, h, j])           // 前半维度
            b = float(in[s, h, j + d_half])  // 后半维度
            out[s, h, j]         = cast<T>(a * cos_phi - b * sin_phi)
            out[s, h, j+d_half]  = cast<T>(b * cos_phi + a * sin_phi)
```

内存布局（连续张量 `[seqlen, nhead, d]`）：
- `in[s, h, k]` 的字节偏移 = `(s * nhead * d + h * d + k) * elem_size`

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/rope/op.cpp`
2. **[NEW]** `src/ops/rope/cpu/rope_cpu.hpp`
3. **[NEW]** `src/ops/rope/cpu/rope_cpu.cpp`

---

### 文件 1：`src/ops/rope/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "rope: pos_ids must be Int64");

    size_t seqlen = in->shape()[0];
    size_t nhead  = in->shape()[1];
    size_t d      = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         seqlen, nhead, d, theta, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         seqlen, nhead, d, theta, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
```

---

### 文件 2：`src/ops/rope/cpu/rope_cpu.hpp`

```cpp
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          size_t seqlen, size_t nhead, size_t d, float theta,
          llaisysDataType_t dtype);
}
```

---

### 文件 3：`src/ops/rope/cpu/rope_cpu.cpp`

```cpp
#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t d_half = d / 2;
    for (size_t s = 0; s < seqlen; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            size_t base = s * nhead * d + h * d;
            for (size_t j = 0; j < d_half; j++) {
                float phi = pos / std::pow(theta, 2.0f * j / static_cast<float>(d));
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                float a = llaisys::utils::cast<float>(in[base + j]);
                float b = llaisys::utils::cast<float>(in[base + j + d_half]);
                out[base + j]        = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[base + j + d_half] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          size_t seqlen, size_t nhead, size_t d, float theta,
          llaisysDataType_t dtype) {
    const int64_t *ids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in), ids,
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in), ids,
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in), ids,
                     seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `d_half = d / 2` | 前半维度 `[0, d_half)` 和后半维度 `[d_half, d)` 配对做旋转 |
| 2 | `pos = float(pos_ids[s])` | 将 int64 位置 id 转为 float 用于频率计算 |
| 3 | `phi = pos / pow(theta, 2j/d)` | 旋转角度，随维度 j 指数衰减（低频到高频） |
| 4 | `cos_phi, sin_phi` | 计算旋转所需的 cos/sin |
| 5 | `a * cos - b * sin` | 前半维度旋转公式 |
| 6 | `b * cos + a * sin` | 后半维度旋转公式 |
| 7 | `base = s * nhead * d + h * d` | 连续张量 `[seqlen, nhead, d]` 中 `[s, h, :]` 的起始偏移 |

## 验证方式

```bash
python test/ops/rope.py
```

预期输出：`Test passed!`
