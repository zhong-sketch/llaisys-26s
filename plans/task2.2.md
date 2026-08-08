# 任务 2.2 — `void embedding(tensor_t out, tensor_t index, tensor_t weight)`

## 任务本质

Embedding 查表操作的本质是：**把一个整数索引数组当作"指针"，从权重矩阵中逐行拷贝对应的行到输出矩阵**。

具体来说：`index` 是 1D 的整数张量（形状 `[N]`），`weight` 是 2D 的嵌入权重矩阵（形状 `[vocab_size, embed_dim]`），`out` 是 2D 输出（形状 `[N, embed_dim]`）。对每个 `i`，执行 `out[i] = weight[index[i]]`，即把 `weight` 的第 `index[i]` 行的数据完整拷贝到 `out` 的第 `i` 行。

这是 LLM 中词元（token）到向量表示的第一步，决定了每个 token 的初始特征。

## 目标

从 `weight`（2D）中复制 `index`（1D）中的行到 `output`（2D）。`index` 必须是 Int64 类型。完成后通过 `test/ops/embedding.py` 的测试。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/ops/embedding/op.hpp` | 函数声明（已存在） |
| `src/ops/embedding/op.cpp` | 调度层，需要实现 |
| `src/ops/embedding/cpu/embedding_cpu.hpp` | CPU 后端声明，**新建** |
| `src/ops/embedding/cpu/embedding_cpu.cpp` | CPU 后端实现，**新建** |

### 测试形状

```python
testShapes = [
    ((1,), (2, 3)),       # idx_shape, embd_shape
    ((50,), (512, 4096)),
]
# out shape = (idx_shape[0], embd_shape[1])
# index 为 i64，值范围 [0, embd_shape[0])
```

### 参数语义

| 参数 | shape | dtype | 说明 |
|------|-------|-------|------|
| `out` | `[N, embed_dim]` | f32/f16/bf16 | 输出张量，写入结果 |
| `index` | `[N]` | i64 | 每个位置要取的行下标 |
| `weight` | `[vocab_size, embed_dim]` | f32/f16/bf16 | 嵌入权重矩阵 |

### 关键计算

每行的字节数 = `embed_dim * elementSize()`，对每个 `i`：

```
memcpy(
    out->data()    + i * embed_dim * elem_size,
    weight->data() + index[i] * embed_dim * elem_size,
    embed_dim * elem_size
)
```

### 参考模式 — argmax op 的两层结构

与 task 2.1 完全相同的"调度层 + CPU 后端"模式。

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/embedding/op.cpp` — 调度层
2. **[NEW]** `src/ops/embedding/cpu/embedding_cpu.hpp` — CPU 后端声明
3. **[NEW]** `src/ops/embedding/cpu/embedding_cpu.cpp` — CPU 后端实现

---

### 文件 1：`src/ops/embedding/op.cpp`

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

    size_t n         = index->numel();          // 行数 N
    size_t embed_dim = weight->shape()[1];       // 每行的元素数
    size_t elem_size = weight->elementSize();    // 每个元素的字节数

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              n, embed_dim, elem_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              n, embed_dim, elem_size);
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

### 文件 2：`src/ops/embedding/cpu/embedding_cpu.hpp`

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

### 文件 3：`src/ops/embedding/cpu/embedding_cpu.cpp`

```cpp
#include "embedding_cpu.hpp"

#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t n, size_t embed_dim, size_t elem_size) {
    const int64_t *idx_ptr  = reinterpret_cast<const int64_t *>(index);
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

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `ASSERT(index->dtype() == LLAISYS_DTYPE_I64)` | index 必须是 Int64（PyTorch 中 int 的默认类型），确保类型安全 |
| 2 | `n = index->numel()` | 要查询的行数 |
| 3 | `embed_dim = weight->shape()[1]` | 每行的元素数量（嵌入维度） |
| 4 | `row_bytes = embed_dim * elem_size` | 每行的字节大小，作为拷贝单位 |
| 5 | `memcpy(out + i*row_bytes, weight + row*row_bytes, row_bytes)` | 逐行查表拷贝，`row = index[i]` 即为 weight 的行下标 |

embedding 的 CPU 实现不需要类型模板，因为数据是整行复制的字节块，与 dtype 无关——dtype 只影响 `elem_size`，而这个信息由调度层通过 `weight->elementSize()` 传入。

## 验证方式

```bash
python test/ops/embedding.py
```

预期输出：`Test passed!`
