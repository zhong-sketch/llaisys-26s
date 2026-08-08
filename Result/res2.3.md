# 结果记录 2.3 — `void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/linear/op.cpp` | **修改** — 调度层 |
| `src/ops/linear/cpu/linear_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/linear/cpu/linear_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/linear/op.cpp`

### 修改前

```cpp
#include "op.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
```

### 修改后（关键部分）

```cpp
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) { CHECK_SAME_DEVICE(out, bias); }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "...");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    const std::byte *bias_ptr = (bias != nullptr) ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           M, N, K, out->dtype());
    }
    // ... device switch ...
}
```

### Diff

```diff
 void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, in, weight);
+    if (bias != nullptr) { CHECK_SAME_DEVICE(out, bias); }
+    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "...");
+    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
+    size_t M = in->shape()[0], K = in->shape()[1], N = weight->shape()[0];
+    const std::byte *bias_ptr = (bias != nullptr) ? bias->data() : nullptr;
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::linear(...);
+    }
+    // switch device...
 }
```

---

## 文件 2：`src/ops/linear/cpu/linear_cpu.hpp`（新建）

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

## 文件 3：`src/ops/linear/cpu/linear_cpu.cpp`（新建）

```cpp
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += cast<float>(in[i*K+k]) * cast<float>(weight[j*K+k]);
            }
            if (bias != nullptr) acc += cast<float>(bias[j]);
            out[i*N+j] = cast<T>(acc);
        }
    }
}
// switch(dtype) 分发 F32/F16/BF16
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `bias != nullptr` 分支校验 | bias 是可选的，`shared_ptr` 为 null 代表无偏置，不需要 device 校验 |
| 2 | `M, K, N` 提取 | `M=in.rows, K=in.cols=weight.cols, N=weight.rows`，对应 `Y[M×N] = X[M×K] × W^T[K×N]` |
| 3 | `bias_ptr = nullptr` | 把 tensor 的可选性转化为 C 指针的空指针，传入 CPU 实现统一处理 |
| 4 | `weight[j*K+k]` | weight 是 `[N,K]`，`weight[j][k]` 正好是转置后的 `W^T[k][j]`，无需实际转置 |
| 5 | `float acc` 累加 | fp16/bf16 精度低，统一用 float 做中间累加，避免累计误差超过测试容忍度 |
| 6 | `cast<T>(acc)` | 累加完成后转回目标类型存储到输出张量 |

## 与前两个 op 的对比

| | argmax (2.1) | embedding (2.2) | linear (2.3) |
|--|--|--|--|
| 核心操作 | 遍历比较 | memcpy 行拷贝 | 三重循环矩阵乘 |
| 类型模板 | 需要（比较需要类型） | 不需要（字节拷贝） | 需要（乘法需要类型） |
| 可选参数 | 无 | 无 | bias 可为 nullptr |
| 中间类型 | float（比较） | 无 | float（累加） |
