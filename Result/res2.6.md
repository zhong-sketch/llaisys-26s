# 结果记录 2.6 — `void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/self_attention/op.cpp` | **修改** — 调度层 |
| `src/ops/self_attention/cpu/self_attention_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/self_attention/cpu/self_attention_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/self_attention/op.cpp`

### Diff

```diff
 void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(attn_val, q, k, v);
+    ASSERT(all->isContiguous(), "...");
+    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
+    size_t qlen = q->shape()[0], nh = q->shape()[1], d = q->shape()[2];
+    size_t kvlen = k->shape()[0], nkvh = k->shape()[1], dv = v->shape()[2];
+    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::self_attention(...);
+    }
+    // ...device switch...
 }
```

---

## 文件 2：`src/ops/self_attention/cpu/self_attention_cpu.hpp`（新建）

```cpp
namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t qlen, size_t nh, size_t kvlen, size_t nkvh,
                    size_t d, size_t dv, float scale, llaisysDataType_t dtype);
}
```

---

## 文件 3：`src/ops/self_attention/cpu/self_attention_cpu.cpp`（新建）

完整三步计算（含 GQA + causal mask + safe-softmax）：

```cpp
template <typename T>
void self_attention_(...) {
    std::vector<float> attn_scores(qlen * kvlen);
    for (size_t h = 0; h < nh; h++) {
        size_t h_kv = h * nkvh / nh;  // GQA 映射

        // Step 1: Q×K^T × scale + causal mask
        for i, j:
            dot = sum_k q[i,h,k] * k[j,h_kv,k]
            masked = (j > kvlen - qlen + i)
            A[i,j] = masked ? -inf : dot * scale

        // Step 2: safe-softmax
        for i:
            max_val = max(A[i,:])
            A[i,:] = exp(A[i,:] - max_val)
            A[i,:] /= sum(A[i,:])

        // Step 3: A × V
        for i, dvi:
            out[i,h,dvi] = sum_j A[i,j] * v[j,h_kv,dvi]
    }
}
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| GQA | `h_kv = h * nkvh / nh` | 每 `nh/nkvh` 个 q-head 共享一个 kv-head，整除映射 |
| Causal mask | `j > kvlen - qlen + i` | q 的第 i 行全局位置是 `kvlen-qlen+i`，不能看到更后面的 k |
| Safe-softmax | `max_val` 平移 | 标准数值稳定写法，避免 exp(large) 溢出为 inf |
| 辅助缓冲区 | `vector<float> attn_scores` | 存放 `[qlen×kvlen]` 的 float 分数矩阵，每个 head 复用 |
| 输出布局 | `attn_val[i*nh*dv + h*dv + dvi]` | 写回连续张量 `[qlen, nh, dv]` 的正确偏移 |

## 与其他 op 的对比

| op | 核心计算 | 特殊处理 |
|----|----------|----------|
| linear (2.3) | 单次矩阵乘 | 可选 bias |
| rms_norm (2.4) | 按行归一化 | 两遍扫描（均方 + 缩放） |
| rope (2.5) | 按 head 做旋转 | 位置编码 |
| self_attention (2.6) | QK^T + softmax + AV | GQA + causal mask + 三阶段计算 |
