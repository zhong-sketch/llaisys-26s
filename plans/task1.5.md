# 任务 1.5 — `tensor_t slice(size_t dim, size_t start, size_t end) const`

## 任务本质

`slice` 的本质是**在某一个维度上截取一段子范围，而不复制数据**。它通过两个手段实现零拷贝：

1. **修改 shape**：把被切片的那个维度从原来的大小改为 `end - start`；
2. **调整 offset**：把张量的起始指针向后偏移 `start * strides[dim]` 个元素，使得新张量的第 0 个元素正好对应原张量在该维度上的第 `start` 个元素。

其他维度的 shape 和所有维度的 strides 完全不变——这正是 stride-based 内存布局的强大之处：通过调整 offset 和 shape，就能从同一块内存中"切出"任意子区域。

## 目标

创建一个新张量，沿给定维度，以 `start`（包含）和 `end`（不包含）索引对原始张量进行切片操作。不涉及数据复制。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/tensor/tensor.cpp` | 需要修改的实现文件，`slice()` 在第 228-231 行 |
| `src/tensor/tensor.hpp` | `TensorMeta`（shape/strides）和 `Tensor` 类声明，含 `_offset` 成员 |
| `src/utils/check.hpp` | 提供 `CHECK_ARGUMENT(condition, message)` 用于参数校验 |

### 关键数据结构

```cpp
class Tensor {
private:
    TensorMeta _meta;       // dtype, shape, strides
    core::storage_t _storage;
    size_t _offset;         // 元素级偏移量（相对 _storage 起始，以字节为单位）
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);
};
```

`_offset` 存储的是**字节偏移量**（`data()` 返回 `_storage->memory() + _offset`），因此计算 offset 时需要将元素偏移乘以 `elementSize()`。

### 参考模式 — `permute()` 中共享 storage 与 offset 的写法

```cpp
// permute() 传入 _offset 保留偏移，共享 _storage
TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
```

`slice()` 采用相同模式，区别在于 offset 需要在 `_offset` 基础上增加切片偏移。

## 实现计划

### 修改位置

- **文件**: `src/tensor/tensor.cpp`
- **行**: 228-231

### 实现代码

```cpp
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 校验 dim 不越界
    CHECK_ARGUMENT(dim < _meta.shape.size(),
                   "slice() dim out of range");

    // 2. 校验 start/end 合法
    CHECK_ARGUMENT(start < end,
                   "slice() start must be less than end");
    CHECK_ARGUMENT(end <= _meta.shape[dim],
                   "slice() end out of range for this dimension");

    // 3. 构建新 shape：只修改被切片的那个维度
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 4. 计算新的字节偏移：在原 _offset 基础上加上 start 对应的偏移
    size_t new_offset = _offset +
                        static_cast<size_t>(_meta.strides[dim]) * start * elementSize();

    // 5. strides 不变，直接复用 _meta
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `dim < shape.size()` | 维度下标不能越界 |
| 2 | `start < end` 且 `end <= shape[dim]` | 切片范围必须合法：非空区间，且不超过该维度的边界 |
| 3 | `new_shape[dim] = end - start` | 只修改被切片维度的大小，其余维度保持不变 |
| 4 | `new_offset = _offset + strides[dim] * start * elementSize()` | 新张量的起始位置向后偏移 `start` 个元素（`strides[dim]` 是元素级步长，再乘以 `elementSize()` 换算成字节） |
| 5 | strides 不变 | 切片不改变各维度间的内存间隔，所有 strides 继续有效 |

### 示例

**shape=[3,4,5]，strides=[20,5,1]，elementSize=4（float）**

调用 `slice(1, 1, 3)`（沿第1维取 [1,3)）：

| | 原始 | slice 后 |
|--|------|----------|
| shape | [3, 4, 5] | [3, **2**, 5] |
| strides | [20, 5, 1] | [20, 5, 1]（不变） |
| offset | 0字节 | 0 + 5×1×4 = **20字节** |

## 验证方式

- 编译项目确认无语法错误
- 验证 `slice(dim, start, end)` 返回正确的 shape
- 验证 `dim` 越界、`start >= end`、`end > shape[dim]` 时抛出异常
- 验证切片后数据内容与原张量对应范围一致（通过 `debug()` 打印）
