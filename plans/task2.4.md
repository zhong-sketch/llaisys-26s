# 任务 2.4 — `void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps)`

## 任务本质

RMS Normalization（均方根归一化）是 LLaMA 等现代 LLM 中替代 LayerNorm 的归一化方式。它的本质是：**对输入张量的每一行，计算该行的均方根（RMS），然后将每个元素除以 RMS，最后乘以可学习的缩放权重**。

与 LayerNorm 不同，RMSNorm 省去了均值中心化（不减去均值），只做缩放，计算量更小，效果相当。

数学公式（对每行 i）：

$$Y_i = \frac{W_i \times X_i}{\sqrt{\frac{1}{d}\left(\sum_{j=1}^{d} X_j^2\right) + \epsilon}}$$

等价写法（与测试脚本 `torch_rms_norm` 一致）：

```python
rms_sq = mean(x^2, dim=-1, keepdim=True) + eps   # 均方 + epsilon
scale  = rsqrt(rms_sq)                            # 1 / sqrt(均方 + eps)
out    = x * scale * w                            # 逐元素缩放
```

## 目标

为输入张量 `in` 的每一行计算：
- `out`：输出 Y，2D 连续张量
- `in`（input）：输入 X，2D 连续张量，沿最后一维（每行，长度 d）归一化
- `weight`：权重 W，1D 张量，与输入张量的一行长度相同
- `eps`：小值 ε 以避免除以零

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/ops/rms_norm/op.hpp` | 函数声明（已存在） |
| `src/ops/rms_norm/op.cpp` | 调度层，需要实现 |
| `src/ops/rms_norm/cpu/rms_norm_cpu.hpp` | CPU 后端声明，**新建** |
| `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` | CPU 后端实现，**新建** |

### 测试规格

```python
testShapes = [(1, 4), (512, 4096)]  # (rows M, cols d)
# w shape = (shape[1],) = (d,)
# eps = 1e-5
```

### 参数维度

| 参数 | shape | 说明 |
|------|-------|------|
| `out` | `[M, d]` | 输出张量 |
| `in` | `[M, d]` | 输入张量，按行归一化 |
| `weight` | `[d]` | 可学习缩放权重（逐元素） |
| `eps` | float | 防止分母为零 |

### 关键计算（对每行 i）

```
sum_sq = 0
for k in 0..d: sum_sq += float(in[i*d+k])^2
rms_scale = 1.0 / sqrt(sum_sq / d + eps)     // rsqrt
for k in 0..d:
    out[i*d+k] = cast<T>(float(in[i*d+k]) * rms_scale * float(weight[k]))
```

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/rms_norm/op.cpp`
2. **[NEW]** `src/ops/rms_norm/cpu/rms_norm_cpu.hpp`
3. **[NEW]** `src/ops/rms_norm/cpu/rms_norm_cpu.cpp`

---

### 文件 1：`src/ops/rms_norm/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];  // 行数
    size_t d = in->shape()[1];  // 每行的元素数（归一化维度）

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             M, d, eps, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             M, d, eps, out->dtype());
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

### 文件 2：`src/ops/rms_norm/cpu/rms_norm_cpu.hpp`

```cpp
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t d, float eps, llaisysDataType_t dtype);
}
```

---

### 文件 3：`src/ops/rms_norm/cpu/rms_norm_cpu.cpp`

```cpp
#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; i++) {
        // 1. 计算该行的均方
        float sum_sq = 0.0f;
        for (size_t k = 0; k < d; k++) {
            float v = llaisys::utils::cast<float>(in[i * d + k]);
            sum_sq += v * v;
        }
        // 2. rsqrt(mean(x^2) + eps)
        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 3. 逐元素缩放并乘以权重
        for (size_t k = 0; k < d; k++) {
            float v = llaisys::utils::cast<float>(in[i * d + k]);
            float w = llaisys::utils::cast<float>(weight[k]);
            out[i * d + k] = llaisys::utils::cast<T>(v * scale * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t d, float eps, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), M, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `sum_sq += v * v` | 计算每行所有元素的平方和（用 float 避免 fp16/bf16 精度丢失） |
| 2 | `1.0f / sqrt(sum_sq/d + eps)` | 计算 rsqrt（倒数平方根），eps 防止除以零 |
| 3 | `v * scale * w` | 逐元素：原值 × 归一化因子 × 可学习权重 |
| 4 | `cast<T>(...)` | 结果转回目标类型存入输出张量 |

## 验证方式

```bash
python test/ops/rms_norm.py
```

预期输出：`Test passed!`
