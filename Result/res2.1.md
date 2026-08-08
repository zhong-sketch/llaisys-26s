# 结果记录 2.1 — `void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/argmax/op.cpp` | **修改** — 实现调度层 |
| `src/ops/argmax/cpu/argmax_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/argmax/cpu/argmax_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/argmax/op.cpp`

### 修改前

```cpp
#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
```

### 修改后

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->isContiguous(), "argmax: vals must be contiguous");

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(),
                           vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(),
                           vals->data(), vals->dtype(), vals->numel());
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

### Diff

```diff
 #include "op.hpp"
+
+#include "../../core/llaisys_core.hpp"
+#include "../../utils.hpp"
+#include "cpu/argmax_cpu.hpp"

 namespace llaisys::ops {
 void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(max_idx, max_val, vals);
+    ASSERT(vals->isContiguous(), "argmax: vals must be contiguous");
+
+    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::argmax(max_idx->data(), max_val->data(),
+                           vals->data(), vals->dtype(), vals->numel());
+    }
+    // ... device dispatch ...
 }
```

---

## 文件 2：`src/ops/argmax/cpu/argmax_cpu.hpp`（新建）

```cpp
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val,
            const std::byte *vals, llaisysDataType_t type, size_t numel);
}
```

---

## 文件 3：`src/ops/argmax/cpu/argmax_cpu.cpp`（新建）

```cpp
#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cstdint>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    float best = std::numeric_limits<float>::lowest();
    int64_t best_idx = 0;
    for (size_t i = 0; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best) {
            best = v;
            best_idx = static_cast<int64_t>(i);
        }
    }
    *max_idx = best_idx;
    *max_val = llaisys::utils::cast<T>(best);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val,
            const std::byte *vals, llaisysDataType_t type, size_t numel) {
    auto *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
```

## 逐行解释

| 层 | 关键点 | 为什么 |
|----|--------|--------|
| 调度层 | `CHECK_SAME_DEVICE` | 三个张量必须在同一设备，否则无法操作 |
| 调度层 | `ASSERT(isContiguous)` | argmax 以连续方式遍历数组，非连续张量 stride 不为1时结果错误 |
| 调度层 | CPU 快路径 | 与 `add/op.cpp` 一致，避免 CPU 设备时重复调用 `setDevice` |
| CPU 实现 | `float best` 中间变量 | fp16/bf16 没有直接比较运算符，统一转为 float 进行大小比较 |
| CPU 实现 | `cast<T>(best)` | 将找到的最大值转回原始类型 T 存入 `max_val` 输出张量 |
| CPU 实现 | `int64_t *idx_ptr` | `max_idx` 张量固定为 i64 类型（对应测试中 `zero_tensor((1,), "i64")`） |

## 设计模式参考

与 `add` op 完全一致的两层结构：

```
ops/argmax/op.cpp          ← 校验 + 设备派发
ops/argmax/cpu/argmax_cpu.cpp  ← CPU 实现（模板 + switch）
```
