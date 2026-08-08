# 任务 1.4 — `tensor_t permute(const std::vector<size_t> &order) const`

## 任务本质

`permute` 的本质是**维度重排**——它改变你"看待"张量各个维度的顺序，但不移动任何数据。实现上，只需要按照 `order` 中指定的下标顺序，重新排列 `shape` 和 `strides` 数组即可。

以转置为例：一个 shape=[3, 4] 的矩阵，strides=[4, 1]，调用 `permute({1, 0})` 后，shape 变为 [4, 3]，strides 变为 [1, 4]。内存中数据完全没有动，只是把"第0维"和"第1维"的解释互换了。

这也解释了为什么 `permute` 之后的张量通常**不再连续**（strides 不再是行优先单调递减序列），需要 `contiguous()` 才能恢复连续布局。

## 目标

创建一个新张量，改变原始张量维度的顺序。转置可以通过这个函数实现，而无需移动数据。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/tensor/tensor.cpp` | 需要修改的实现文件，`permute()` 在第 177-180 行 |
| `src/tensor/tensor.hpp` | `TensorMeta`（shape/strides）和 `Tensor` 类声明 |
| `src/utils/check.hpp` | 提供 `CHECK_ARGUMENT(condition, message)` 用于参数校验 |

### 关键数据结构

```cpp
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;        // 各维度大小
    std::vector<ptrdiff_t> strides;   // 各维度步长（以元素为单位）
};
```

### 关键约束

`order` 参数必须满足：
1. 长度等于张量的维度数 `ndim()`
2. 是 `[0, ndim)` 的一个**排列**（每个下标恰好出现一次，没有重复、没有越界）

### 参考模式 — `view()` 的 TensorMeta 构造与 shared_ptr 创建

```cpp
// view() 中构造新张量（共享 _storage，保留 _offset）
TensorMeta new_meta{_meta.dtype, shape, new_strides};
return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
```

`permute()` 采用完全相同的模式，区别仅在于 shape 和 strides 的生成方式。

## 实现计划

### 修改位置

- **文件**: `src/tensor/tensor.cpp`
- **行**: 177-180

### 实现代码

```cpp
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = _meta.shape.size();

    // 1. 校验 order 长度
    CHECK_ARGUMENT(order.size() == ndim_,
                   "permute() order size must match tensor ndim");

    // 2. 校验 order 是合法排列（无越界、无重复）
    std::vector<bool> seen(ndim_, false);
    for (size_t idx : order) {
        CHECK_ARGUMENT(idx < ndim_, "permute() order index out of range");
        CHECK_ARGUMENT(!seen[idx], "permute() order contains duplicate index");
        seen[idx] = true;
    }

    // 3. 按 order 重排 shape 和 strides
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    // 4. 构建新 TensorMeta，共享同一 _storage
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `order.size() == ndim_` | `order` 长度必须等于张量维度数，否则无法一一对应 |
| 2 | `seen` 数组校验 | 检查每个 `order[i]` 在 `[0, ndim)` 范围内且不重复，确保是合法排列 |
| 3 | `new_shape[i] = _meta.shape[order[i]]` | 新的第 i 维 = 原来的第 order[i] 维，shape 和 strides 同步重排 |
| 4 | `new Tensor(new_meta, _storage, _offset)` | 共享 `_storage`（零拷贝），保留 `_offset`，只换元数据 |

### 示例

**转置 shape=[3,4]，order={1,0}：**

| | 原始 | permute 后 |
|--|------|------------|
| shape | [3, 4] | [4, 3] |
| strides | [4, 1] | [1, 4] |
| 数据 | 不动 | 不动 |

**三维张量 shape=[2,3,4]，order={2,0,1}（将 dim2 移到最前）：**

| | 原始 | permute 后 |
|--|------|------------|
| shape | [2, 3, 4] | [4, 2, 3] |
| strides | [12, 4, 1] | [1, 12, 4] |

## 验证方式

- 编译项目确认无语法错误
- 验证 `permute({1,0})` 实现正确的矩阵转置效果
- 验证 `permute` 后 `isContiguous()` 返回 false（步长不再是行优先单调序列）
- 验证非法 order（越界/重复/长度错误）时抛出异常
