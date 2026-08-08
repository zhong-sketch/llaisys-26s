# 结果记录 2.4 — `void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/rms_norm/op.cpp` | **修改** — 调度层 |
| `src/ops/rms_norm/cpu/rms_norm_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/rms_norm/op.cpp`

### Diff

```diff
 void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, in, weight);
+    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "...");
+    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
+    size_t M = in->shape()[0];  // 行数
+    size_t d = in->shape()[1];  // 归一化维度
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::rms_norm(out->data(), in->data(), weight->data(), M, d, eps, out->dtype());
+    }
+    // ...device switch...
 }
```

---

## 文件 2：`src/ops/rms_norm/cpu/rms_norm_cpu.hpp`（新建）

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

## 文件 3：`src/ops/rms_norm/cpu/rms_norm_cpu.cpp`（新建）

```cpp
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; i++) {
        // 1. 计算均方
        float sum_sq = 0.0f;
        for (size_t k = 0; k < d; k++) {
            float v = cast<float>(in[i*d+k]);
            sum_sq += v * v;
        }
        // 2. rsqrt
        float scale = 1.0f / sqrt(sum_sq / d + eps);
        // 3. 缩放 + 权重
        for (size_t k = 0; k < d; k++) {
            float v = cast<float>(in[i*d+k]);
            float w = cast<float>(weight[k]);
            out[i*d+k] = cast<T>(v * scale * w);
        }
    }
}
// switch(dtype) 分发 F32/F16/BF16
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `sum_sq += v*v` | 逐元素平方求和，用 float 累加避免 fp16 溢出 |
| 2 | `scale = 1/sqrt(sum_sq/d + eps)` | RMS 的倒数，eps 防止分母为零 |
| 3 | `v * scale * w` | 先归一化（×scale），再乘以可学习权重 w |
| 4 | `cast<T>(...)` | 最终结果从 float 转回原始类型 T |

## 与其他 op 的对比

| | linear (2.3) | rms_norm (2.4) |
|--|--|--|
| 核心操作 | 行×列点积（矩阵乘） | 按行求均方根然后缩放 |
| 两次遍历 | 无（单次内积循环） | 有（先求 sum_sq，再缩放） |
| 权重作用 | 左乘（矩阵乘） | 逐元素（Hadamard 积） |
| 中间类型 | float（点积累加） | float（平方和累加） |
