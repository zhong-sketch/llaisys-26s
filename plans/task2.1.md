# 任务 2.1 — `void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals)`

## 任务本质

argmax 就是**在一个 1D 张量中找最大值及其下标**，并把结果分别写入两个输出张量 `max_val`（最大值）和 `max_idx`（最大值的索引）。

从工程角度看，这是第一个需要实现完整"调度 → CPU 实现"两层结构的 op：
- **调度层**（`op.cpp`）：校验输入、根据设备类型派发到对应后端
- **CPU 实现层**（`cpu/argmax_cpu.cpp`）：用模板处理各种数据类型，执行实际的遍历比较

这个 op 是后续更复杂 op（如 attention 中的 softmax）的基础模式，掌握它就掌握了整个 op 框架的开发范式。

## 目标

获取张量 `vals` 的最大值及其索引，分别存储在 `max_val` 和 `max_idx` 中。暂时假设 `vals` 是 1D 张量，`max_idx` 和 `max_val` 是包含单个元素的 1D 张量。完成后通过 `test/ops/argmax.py` 的测试。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/ops/argmax/op.hpp` | 函数声明（已存在） |
| `src/ops/argmax/op.cpp` | 调度层，需要实现：校验 + 按设备派发 |
| `src/ops/argmax/cpu/argmax_cpu.hpp` | CPU 后端声明，**新建** |
| `src/ops/argmax/cpu/argmax_cpu.cpp` | CPU 后端实现，**新建** |

### 测试内容

```python
# test/ops/argmax.py
# vals 为 1D 张量，形状 (4,) 或 (4096,)，dtype = f32/f16/bf16
torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))
llaisys.Ops.argmax(max_idx_, max_val_, vals_)
# 验证 max_val_ == max_val 或 max_idx_ == max_idx
```

### 参考模式 — `add` op 的完整实现

**调度层 `add/op.cpp`：**
```cpp
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    // ...校验...
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }
    // ...其他设备...
}
```

**CPU 实现 `add/cpu/add_cpu.cpp`：**
```cpp
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) { c[i] = a[i] + b[i]; }
}
void add(std::byte *c, ...) {
    switch (type) {
    case LLAISYS_DTYPE_F32: return add_(...); // 类型分发
    // ...
    }
}
```

### 关键 API

```cpp
// 类型转换（处理 fp16/bf16）
utils::cast<float>(val);        // 转为 float 进行比较
utils::cast<T>(float_val);      // 转回目标类型存储

// 张量访问
vals->data()                    // 返回 std::byte* 原始指针
vals->numel()                   // 元素数量
vals->dtype()                   // 数据类型
vals->deviceType()              // 设备类型
```

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/argmax/op.cpp` — 实现调度层
2. **[NEW]** `src/ops/argmax/cpu/argmax_cpu.hpp` — CPU 后端声明
3. **[NEW]** `src/ops/argmax/cpu/argmax_cpu.cpp` — CPU 后端实现

---

### 文件 1：`src/ops/argmax/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->isContiguous(), "argmax: vals must be contiguous");

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

---

### 文件 2：`src/ops/argmax/cpu/argmax_cpu.hpp`

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

### 文件 3：`src/ops/argmax/cpu/argmax_cpu.cpp`

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
        return argmax_(idx_ptr,
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr,
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr,
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
```

### 逻辑拆解

| 步骤 | 说明 |
|------|------|
| `argmax_<T>` 模板 | 统一用 `float` 中间变量做比较（处理 fp16/bf16 无法直接比较的问题），记录最大值及其 index |
| `*max_val = cast<T>(best)` | 把 float 的最大值转回原始类型 T 存入输出张量 |
| `switch(type)` 分发 | 根据 dtype 做类型转换，调用具体的模板实例 |
| 调度层 CPU 快路径 | 与 `add/op.cpp` 一致，先判断 CPU 直接执行，避免重复设置 context |

## 验证方式

```bash
python test/ops/argmax.py
```

预期输出：`Test passed!`（通过 f32、f16、bf16 在 shape=(4,) 和 (4096,) 上的测试）
