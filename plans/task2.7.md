# 任务 2.7 — `void swiglu(tensor_t out, tensor_t gate, tensor_t up)`

## 任务本质

SwiGLU（Swish-Gated Linear Unit）是现代 LLM（如 LLaMA）FFN 层的激活函数，它替代了传统的 ReLU 或 GeLU。

公式：`out_i = up_i × sigmoid(gate_i) × gate_i = up_i × gate_i / (1 + exp(-gate_i))`

即：**用 gate 过 Swish 激活（= gate × sigmoid(gate)），再逐元素乘以 up**。

这是一个纯逐元素（element-wise）操作，是这批任务中**最简单的 op**：不涉及任何维度间的归约或矩阵乘法，只是每个位置独立计算一个标量公式。

## 目标

逐元素函数，计算：`out_i = up_i × gate_i / (1 + exp(-gate_i))`

- `out`、`up`、`gate` 是具有相同形状 `[seqlen, intermediate_size]` 的 2D 连续张量
- 函数签名：`void swiglu(tensor_t out, tensor_t gate, tensor_t up)`

## 背景分析

### 测试规格

```python
torch_swiglu(out, gate, up):
    out = up * gate / (1 + exp(-gate.float()))

testShapes = [(2, 3), (512, 4096)]
testDtype  = ["f32", "f16", "bf16"]
```

### 关键计算（逐元素）

```
for i in 0..numel:
    g = float(gate[i])
    sigmoid_g = 1.0f / (1.0f + exp(-g))
    out[i] = cast<T>(float(up[i]) * g * sigmoid_g)
```

### 参考模式

与 `add` op 结构完全一致：调度层 + CPU 模板实现 + switch 分发。

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/swiglu/op.cpp`
2. **[NEW]** `src/ops/swiglu/cpu/swiglu_cpu.hpp`
3. **[NEW]** `src/ops/swiglu/cpu/swiglu_cpu.cpp`

---

### 文件 1：`src/ops/swiglu/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), out->numel());
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

### 文件 2：`src/ops/swiglu/cpu/swiglu_cpu.hpp`

```cpp
#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);
}
```

---

### 文件 3：`src/ops/swiglu/cpu/swiglu_cpu.cpp`

```cpp
#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * g * sigmoid_g);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `g = float(gate[i])` | gate 转 float 用于后续数值计算 |
| 2 | `sigmoid_g = 1/(1+exp(-g))` | 计算 Sigmoid 激活值 |
| 3 | `u * g * sigmoid_g` | `up * swish(gate)`，其中 swish(x) = x × sigmoid(x) |
| 4 | `cast<T>(...)` | 结果转回原始类型存储 |

## 验证方式

```bash
python test/ops/swiglu.py
```

预期输出：`Test passed!`
