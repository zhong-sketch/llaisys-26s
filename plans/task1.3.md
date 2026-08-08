# 任务 1.3 — `tensor_t view(const std::vector<size_t> &shape) const`

## 任务本质

张量的 `view` 操作本质上是**在不复制任何数据的前提下，给同一块内存换一个新的"形状解释"**。它只修改元数据（shape 和 strides），而底层的 `_storage` 完全共享。

这要求两个前提成立：
1. **元素总数不变**：新旧 shape 的 `numel()` 必须相等；
2. **张量必须是内存连续的**：如果张量经过了 `permute` 或 `slice`，其 strides 不再是标准行优先步长，无法通过简单重新计算 strides 来重新解释形状。

`view` 不是 `reshape`——它绝不做数据拷贝。如果张量不连续，应该抛出异常而不是静默地返回错误结果。

## 目标

通过拆分或合并原始维度将原始张量重塑为给定形状，不涉及数据传输。要求张量必须连续，且新旧形状的元素总数一致，否则引发错误。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/tensor/tensor.cpp` | 需要修改的实现文件，`view()` 在第 182-185 行 |
| `src/tensor/tensor.hpp` | `TensorMeta`（shape/strides）和 `Tensor` 类声明 |
| `src/utils/check.hpp` | 提供 `CHECK_ARGUMENT(condition, message)` 用于参数校验和错误抛出 |

### 关键 API

```cpp
// 参数校验宏（条件不满足则打印错误并抛出 std::invalid_argument）
CHECK_ARGUMENT(condition, message);

// 已实现的相关方法
bool isContiguous() const;         // 检查张量是否内存连续
size_t numel() const;              // 返回元素总数
```

### 关键数据结构

```cpp
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;        // 各维度大小
    std::vector<ptrdiff_t> strides;   // 各维度步长（以元素为单位）
};
```

### 参考模式 — `create()` 中的步长生成（tensor.cpp:18-24）

`view` 生成新 strides 的逻辑与 `create()` 完全相同（行优先连续步长）：

```cpp
size_t stride = 1;
for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;
    stride *= shape[ndim_ - i];
}
```

## 实现计划

### 修改位置

- **文件**: `src/tensor/tensor.cpp`
- **行**: 182-185

### 实现代码

```cpp
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 张量必须内存连续
    CHECK_ARGUMENT(isContiguous(), "view() requires a contiguous tensor");

    // 2. 新旧 shape 的元素总数必须一致
    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "view() shape is incompatible: element count mismatch");

    // 3. 计算新的行优先连续 strides
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        new_strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    // 4. 构建新 TensorMeta，共享同一 _storage
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `CHECK_ARGUMENT(isContiguous(), ...)` | 非连续张量（经过 permute/slice）不能 view，必须先调用 `contiguous()` 使其连续 |
| 2 | `new_numel == numel()` | 新 shape 的元素总量必须等于原 shape，否则无法对应同一块内存 |
| 3 | strides 计算循环 | 为新 shape 生成标准行优先步长，逻辑与 `create()` 完全一致 |
| 4 | `new Tensor(new_meta, _storage, _offset)` | 共享 `_storage`（不复制数据），保留原始 `_offset`，只替换元数据 |

### 示例

| 原 shape | 原 strides | 连续？ | 新 shape | 结果 |
|----------|-----------|--------|----------|------|
| [2, 3, 5] | [15, 5, 1] | ✅ | [2, 15] | 成功，新 strides = [15, 1] |
| [2, 3, 5] | [30, 10, 1] | ❌ | [2, 15] | 抛出异常（非连续） |
| [2, 3, 5] | [15, 5, 1] | ✅ | [6, 4] | 抛出异常（2×3×5=30 ≠ 6×4=24） |

## 验证方式

- 编译项目确认无语法错误
- 运行测试，验证 shape=[2,3,5] → view([2,15]) 成功
- 验证非连续张量调用 view 时抛出异常
- 验证元素数不匹配时抛出异常
