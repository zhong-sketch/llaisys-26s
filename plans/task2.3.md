# 任务 2.3 — `void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias)`

## 任务本质

Linear（全连接层）的本质是**矩阵乘法加可选偏置**：`Y = X × W^T + b`。

关键注意点：
- **weight 没有预先转置**，所以需要在计算时处理转置——即对 `in` 的每一行与 `weight` 的每一行（而非列）做点积；
- **bias 是可选的**：函数签名中 `bias` 可能为 `nullptr`（对应 Python 侧传 `None`），必须支持不提供偏置的情况；
- 精度要求宽松（f16 容忍 1e-3，bf16 容忍 1e-2），可以用 float 做中间累加。

在 LLM 中，linear 是最频繁调用的算子（每个 attention 层和 FFN 层都有多个），性能至关重要，但此任务先实现正确性。

## 目标

计算 `out = in × weight^T + bias`：
- `out`：输出 Y，2D 连续张量，形状 `[M, N]`
- `in`（input）：输入 X，2D 连续张量，形状 `[M, K]`
- `weight`：权重 W，2D 连续张量，形状 `[N, K]`（**未转置**）
- `bias`（可选）：偏置 b，1D 张量，形状 `[N]`，可为 `nullptr`

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/ops/linear/op.hpp` | 函数声明（已存在） |
| `src/ops/linear/op.cpp` | 调度层，需要实现 |
| `src/ops/linear/cpu/linear_cpu.hpp` | CPU 后端声明，**新建** |
| `src/ops/linear/cpu/linear_cpu.cpp` | CPU 后端实现，**新建** |

### 测试形状

```python
testShapes = [
    # out_shape,    x_shape,      w_shape,       use_bias
    ((2, 3),        (2, 4),       (3, 4),         True),
    ((512, 4096),   (512, 4096),  (4096, 4096),   True),
]
# out[M,N] = in[M,K] × weight[N,K]^T + bias[N]
```

### torch.nn.functional.linear 的语义

```python
torch.nn.functional.linear(x, w, bias, out=out)
# 等价于: out = x @ w.T + bias
```

### 维度关系

- `M` = `in.shape[0]` = `out.shape[0]`（batch 行数）
- `K` = `in.shape[1]` = `weight.shape[1]`（输入特征维）
- `N` = `weight.shape[0]` = `out.shape[0]`（输出特征维）

### 关键计算（无转置）

```
out[i][j] = sum_{k=0}^{K-1} in[i][k] * weight[j][k]  +  bias[j]
```

weight 已经是 `[N, K]`，所以 `weight[j][k]` 就是第 j 行第 k 列，无需做实际的转置操作。

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/linear/op.cpp`
2. **[NEW]** `src/ops/linear/cpu/linear_cpu.hpp`
3. **[NEW]** `src/ops/linear/cpu/linear_cpu.cpp`

---

### 文件 1：`src/ops/linear/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // bias 为可选，不参与 device 校验
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: out/in/weight must be contiguous");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    const std::byte *bias_ptr = (bias != nullptr) ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           M, N, K, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           M, N, K, out->dtype());
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

### 文件 2：`src/ops/linear/cpu/linear_cpu.hpp`

```cpp
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype);
}
```

---

### 文件 3：`src/ops/linear/cpu/linear_cpu.cpp`

```cpp
#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += llaisys::utils::cast<float>(in[i * K + k]) *
                       llaisys::utils::cast<float>(weight[j * K + k]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * N + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `bias != nullptr` 判断 | bias 是可选参数，`tensor_t` 本质是 `shared_ptr`，nullptr 代表不提供偏置 |
| 2 | `M, K, N` 提取 | `M=in.rows, K=in.cols=weight.cols, N=weight.rows` |
| 3 | `in[i*K+k] * weight[j*K+k]` | 因为 weight 是 `[N,K]`，`weight[j][k]` 就是转置后的值，无需额外转置 |
| 4 | `float acc` 中间累加 | fp16/bf16 精度不足，统一用 float 累加，最后 `cast<T>` 回目标类型 |
| 5 | `bias[j]` 加偏置 | bias 是 1D `[N]`，对每个输出列 j 加一个固定偏置 |

## 验证方式

```bash
python test/ops/linear.py
```

预期输出：`Test passed!`（通过 f32/f16/bf16，形状 (2,3)×(2,4)×(3,4) 和 (512,4096)×(512,4096)×(4096,4096) 的测试）
