# 结果记录 2.5 — `void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/rope/op.cpp` | **修改** — 调度层 |
| `src/ops/rope/cpu/rope_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/rope/cpu/rope_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/rope/op.cpp`

### Diff

```diff
 void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, in, pos_ids);
+    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "...");
+    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
+    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be Int64");
+    size_t seqlen = in->shape()[0];
+    size_t nhead  = in->shape()[1];
+    size_t d      = in->shape()[2];
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::rope(out->data(), in->data(), pos_ids->data(),
+                         seqlen, nhead, d, theta, out->dtype());
+    }
+    // ...device switch...
 }
```

---

## 文件 2：`src/ops/rope/cpu/rope_cpu.hpp`（新建）

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

## 文件 3：`src/ops/rope/cpu/rope_cpu.cpp`（新建）

```cpp
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t d_half = d / 2;
    for (size_t s = 0; s < seqlen; s++) {
        float pos = float(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            size_t base = s * nhead * d + h * d;
            for (size_t j = 0; j < d_half; j++) {
                float phi = pos / pow(theta, 2.0f * j / d);
                float a = cast<float>(in[base + j]);
                float b = cast<float>(in[base + j + d_half]);
                out[base + j]          = cast<T>(a * cos(phi) - b * sin(phi));
                out[base + j + d_half] = cast<T>(b * cos(phi) + a * sin(phi));
            }
        }
    }
}
// switch(dtype) → F32/F16/BF16
```

## 逐行解释

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `d_half = d / 2` | 将每个 head 的 d 维向量分为两半：前半 a，后半 b |
| 2 | `pos = float(pos_ids[s])` | 获取第 s 个 token 的全局位置 id（int64 → float） |
| 3 | `base = s*nhead*d + h*d` | 连续张量 `[seqlen, nhead, d]` 中 `[s, h, :]` 起始偏移 |
| 4 | `phi = pos / pow(theta, 2j/d)` | 第 j 个频率分量的旋转角度，低 j 对应低频（大角度变化慢） |
| 5 | `a*cos - b*sin` | 前半维度旋转公式（复平面实部） |
| 6 | `b*cos + a*sin` | 后半维度旋转公式（复平面虚部） |
| 7 | `cast<T>(...)` | float 结果转回原始类型存入 out |

## 与其他 op 的对比

| | rms_norm (2.4) | rope (2.5) |
|--|--|--|
| 遍历结构 | 行×列（两遍） | seqlen×nhead×d_half（一遍） |
| 依赖关系 | 每行先求全局 RMS 再缩放 | 每对 (a,b) 独立旋转 |
| 三角函数 | 无 | cos/sin |
| 位置信息 | 无 | pos_ids[s] 决定旋转角度 |
