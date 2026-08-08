# 结果记录 2.2 — `void embedding(tensor_t out, tensor_t index, tensor_t weight)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/embedding/op.cpp` | **修改** — 实现调度层 |
| `src/ops/embedding/cpu/embedding_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/embedding/cpu/embedding_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/embedding/op.cpp`

### 修改前

```cpp
#include "op.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
```

### 修改后

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "embedding: index must be Int64");

    size_t n         = index->numel();
    size_t embed_dim = weight->shape()[1];
    size_t elem_size = weight->elementSize();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              n, embed_dim, elem_size);
    }
    // ...device dispatch...
}
} // namespace llaisys::ops
```

### Diff

```diff
 #include "op.hpp"
+#include "../../core/llaisys_core.hpp"
+#include "../../utils.hpp"
+#include "cpu/embedding_cpu.hpp"

 namespace llaisys::ops {
 void embedding(tensor_t out, tensor_t index, tensor_t weight) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, index, weight);
+    ASSERT(...isContiguous()..., "...");
+    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "...");
+    size_t n = index->numel();
+    size_t embed_dim = weight->shape()[1];
+    size_t elem_size = weight->elementSize();
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::embedding(...);
+    }
+    // switch device...
 }
```

---

## 文件 2：`src/ops/embedding/cpu/embedding_cpu.hpp`（新建）

```cpp
#pragma once
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size);
}
```

---

## 文件 3：`src/ops/embedding/cpu/embedding_cpu.cpp`（新建）

```cpp
#include "embedding_cpu.hpp"
#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    size_t row_bytes = embed_dim * elem_size;
    for (size_t i = 0; i < n; i++) {
        int64_t row = idx_ptr[i];
        std::memcpy(
            out    + i   * row_bytes,
            weight + row * row_bytes,
            row_bytes
        );
    }
}
} // namespace llaisys::ops::cpu
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `ASSERT(index->dtype() == LLAISYS_DTYPE_I64)` | index 必须是 Int64，PyTorch 中默认整数类型，确保 `reinterpret_cast<int64_t*>` 安全 |
| 2 | `n = index->numel()` | 有多少个 index，就要查多少行 |
| 3 | `embed_dim = weight->shape()[1]` | weight 矩阵每行有多少个元素 |
| 4 | `elem_size = weight->elementSize()` | 每个元素多少字节（随 dtype 变化），调度层统一计算传给 CPU 实现 |
| 5 | `row_bytes = embed_dim * elem_size` | 每行的字节数，作为 memcpy 的基本单位 |
| 6 | `row = idx_ptr[i]` | 读取第 i 个整数索引，即 weight 矩阵中要取的行号 |
| 7 | `memcpy(out + i*row_bytes, weight + row*row_bytes, row_bytes)` | 把 weight 的第 row 行完整复制到 out 的第 i 行 |

## 与 argmax 的对比

| | argmax (task 2.1) | embedding (task 2.2) |
|--|--|--|
| CPU 实现 | 需要类型模板（比较大小依赖类型） | 不需要模板（整行字节拷贝，dtype 无关） |
| 核心操作 | 遍历 + 比较 | 遍历 + memcpy |
| 输出 | 单个标量（最大值 + 下标） | 多行向量（查表结果） |
