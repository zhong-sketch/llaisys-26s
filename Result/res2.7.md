# 结果记录 2.7 — `void swiglu(tensor_t out, tensor_t gate, tensor_t up)`

## 修改/新建文件

| 文件 | 操作 |
|------|------|
| `src/ops/swiglu/op.cpp` | **修改** — 调度层 |
| `src/ops/swiglu/cpu/swiglu_cpu.hpp` | **新建** — CPU 后端声明 |
| `src/ops/swiglu/cpu/swiglu_cpu.cpp` | **新建** — CPU 后端实现 |

---

## 文件 1：`src/ops/swiglu/op.cpp`

### Diff

```diff
 void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
-    TO_BE_IMPLEMENTED();
+    CHECK_SAME_DEVICE(out, gate, up);
+    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
+    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
+    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "...");
+    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
+        return cpu::swiglu(out->data(), gate->data(), up->data(),
+                           out->dtype(), out->numel());
+    }
+    // ...device switch...
 }
```

---

## 文件 2：`src/ops/swiglu/cpu/swiglu_cpu.hpp`（新建）

```cpp
namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);
}
```

---

## 文件 3：`src/ops/swiglu/cpu/swiglu_cpu.cpp`（新建）

```cpp
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = cast<float>(gate[i]);
        float u = cast<float>(up[i]);
        float sigmoid_g = 1.0f / (1.0f + exp(-g));
        out[i] = cast<T>(u * g * sigmoid_g);
    }
}
// switch(dtype) → F32/F16/BF16
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `g = cast<float>(gate[i])` | gate 转 float，避免 fp16/bf16 精度不足 |
| 2 | `sigmoid_g = 1/(1+exp(-g))` | Sigmoid 函数，将 gate 映射到 (0,1) |
| 3 | `u * g * sigmoid_g` | Swish(gate) = gate × sigmoid(gate)，再乘以 up |
| 4 | `cast<T>(...)` | float 结果转回原始类型存储 |

## 与其他 op 的对比

| op | 复杂度 | 核心特性 |
|----|--------|----------|
| add (参考) | 逐元素 | 单纯加法 |
| swiglu (2.7) | 逐元素 | sigmoid + 乘法（非线性激活） |
| rms_norm (2.4) | 行归约 | 需要两遍扫描 |
| self_attention (2.6) | 矩阵 | 三阶段计算 |

SwiGLU 是本批次实现难度最低的 op，结构与 `add` 完全一致，仅计算公式不同。
