# 任务 2.6 — `void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale)`

## 任务本质

自注意力（Self-Attention）是 Transformer 的核心算子，本质是：**Q 向 K 查询相关性得到注意力权重，再用权重加权 V 的值**。

整个计算分两步：
1. **注意力分数**：`A = Q × K^T × scale`，再加 causal mask（下三角，未来位置置为 -inf），再做 softmax
2. **加权求和**：`output = softmax(A) × V`

额外复杂性：
- 输入张量形状是 `[seqlen, nhead, d]`（seq-first），但矩阵乘需要 head-first `[nhead, seqlen, d]`，所以要逻辑上做 transpose
- 支持 **GQA（Grouped Query Attention）**：`nhead != nkvh`，key/value 的 head 数可以少于 query，需要重复扩展（每个 kv-head 对应 `nhead/nkvh` 个 q-head）
- Causal mask：下三角矩阵，`A[i][j]` 只在 `j <= i + (kvlen - qlen)` 时有效（处理 prefill 和 decode 两种情况）

## 目标

为查询张量 q、键张量 k 和值张量 v 计算自注意力。

- `attn_val`：结果注意力值张量，形状 `[seqlen, nhead, dv]`，连续
- `q`：查询张量，形状 `[seqlen, nhead, d]`，连续
- `k`：键张量，形状 `[total_len, nkvhead, d]`，连续
- `v`：值张量，形状 `[total_len, nkvhead, dv]`，连续
- `scale`：缩放因子，通常 `1/sqrt(d)`

## 背景分析

### 测试规格

```python
testShapes = [
    # qlen, kvlen, nh, nkvh, hd
    (2, 2, 1, 1, 4),       # 基础：多 q-head = kv-head
    (5, 11, 4, 2, 8),      # GQA：nh=4, nkvh=2，prefill（qlen < kvlen）
]
# q shape = (qlen, nh, hd)
# k, v shape = (kvlen, nkvh, hd)
# attn_val shape = (qlen, nh, hd)
```

### torch_self_attention 等价实现

```python
# 1. 转置到 head-first [nhead, qlen/kvlen, d]
q = q.transpose(-2, -3)        # [nh, qlen, d]
k = k.transpose(-2, -3)        # [nkvh, kvlen, d]
v = v.transpose(-2, -3)        # [nkvh, kvlen, dv]

# 2. GQA 扩展：把 k/v 重复 nh/nkvh 次
k = k.repeat_interleave(nh // nkvh, -3)   # [nh, kvlen, d]
v = v.repeat_interleave(nh // nkvh, -3)   # [nh, kvlen, dv]

# 3. Causal mask [qlen, kvlen]
mask = tril(ones(qlen, kvlen), diagonal=kvlen-qlen)  # True=有效，False=-inf

# 4. Attention scores
A = q @ k^T * scale            # [nh, qlen, kvlen]
A += mask_bias                  # 未来位置加 -inf
A = softmax(A, dim=-1)         # [nh, qlen, kvlen]

# 5. 加权求和
out = A @ v                     # [nh, qlen, dv]
attn_val = out.transpose(-2,-3) # [qlen, nh, dv]
```

### 关键内存布局（连续张量）

- `q[s, h, k]` 偏移 = `(s * nh * d + h * d + k)`
- `k_tensor[t, h, k]` 偏移 = `(t * nkvh * d + h * d + k)`

访问 "第 h_q 个 q-head 与第 h_kv 个 kv-head（GQA 映射）的关系"：
- `h_kv = h_q * nkvh / nh`（整除）

## 实现计划

### 修改/新建文件

1. **[MODIFY]** `src/ops/self_attention/op.cpp`
2. **[NEW]** `src/ops/self_attention/cpu/self_attention_cpu.hpp`
3. **[NEW]** `src/ops/self_attention/cpu/self_attention_cpu.cpp`

---

### 文件 1：`src/ops/self_attention/op.cpp`

```cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
           k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t qlen  = q->shape()[0];
    size_t nh    = q->shape()[1];
    size_t d     = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh  = k->shape()[1];
    size_t dv    = v->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            qlen, nh, kvlen, nkvh, d, dv, scale, attn_val->dtype());
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

### 文件 2：`src/ops/self_attention/cpu/self_attention_cpu.hpp`

```cpp
#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                    size_t d, size_t dv, float scale,
                    llaisysDataType_t dtype);
}
```

---

### 文件 3：`src/ops/self_attention/cpu/self_attention_cpu.cpp`

核心模板函数，按 head 循环执行完整的 QK^T + causal_mask + softmax + @V：

```cpp
template <typename T>
void self_attention_(T *attn_val,
                     const T *q, const T *k, const T *v,
                     size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                     size_t d, size_t dv, float scale) {
    // 辅助缓冲区：每个 head 的注意力分数矩阵 [qlen × kvlen]
    std::vector<float> attn_scores(qlen * kvlen);

    for (size_t h = 0; h < nh; h++) {
        size_t h_kv = h * nkvh / nh;  // GQA：映射到对应的 kv-head

        // ---- Step 1: A = Q_h × K_hkv^T × scale ----
        for (size_t i = 0; i < qlen; i++) {
            for (size_t j = 0; j < kvlen; j++) {
                float dot = 0.0f;
                for (size_t dk = 0; dk < d; dk++) {
                    float qv = cast<float>(q[i * nh * d + h * d + dk]);
                    float kv = cast<float>(k[j * nkvh * d + h_kv * d + dk]);
                    dot += qv * kv;
                }
                // Causal mask: i 对应的全局位置是 kvlen-qlen+i
                bool masked = (j > kvlen - qlen + i);
                attn_scores[i * kvlen + j] = masked ? -INFINITY : dot * scale;
            }
        }

        // ---- Step 2: Softmax over last dim ----
        for (size_t i = 0; i < qlen; i++) {
            float max_val = -INFINITY;
            for (size_t j = 0; j < kvlen; j++)
                max_val = std::max(max_val, attn_scores[i*kvlen+j]);
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[i*kvlen+j] = std::exp(attn_scores[i*kvlen+j] - max_val);
                sum += attn_scores[i*kvlen+j];
            }
            for (size_t j = 0; j < kvlen; j++)
                attn_scores[i*kvlen+j] /= sum;
        }

        // ---- Step 3: out_h = attn_scores × V_hkv ----
        for (size_t i = 0; i < qlen; i++) {
            for (size_t dvi = 0; dvi < dv; dvi++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    acc += attn_scores[i*kvlen+j] *
                           cast<float>(v[j * nkvh * dv + h_kv * dv + dvi]);
                }
                attn_val[i * nh * dv + h * dv + dvi] = cast<T>(acc);
            }
        }
    }
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| GQA 映射 | `h_kv = h * nkvh / nh` | 将 q-head 映射到对应的 kv-head（整除，每 nh/nkvh 个 q 共享一个 kv） |
| Causal mask | `j > kvlen - qlen + i` | 位置关系：q 的第 i 行对应全局位置 `kvlen-qlen+i`，只能看到 `j <= ...` 的 k |
| Numerically stable softmax | `max_val` 平移 | 避免 exp 溢出，标准 safe-softmax 写法 |
| 矩阵乘 | 三重循环 | Q×K^T 和 A×V 都用简单三重循环，正确性优先 |
| 输出布局 | `attn_val[i * nh * dv + h * dv + dvi]` | 结果写回 `[qlen, nh, dv]` 连续布局 |

## 验证方式

```bash
python test/ops/self_attention.py
```

预期输出：`Test passed!`
