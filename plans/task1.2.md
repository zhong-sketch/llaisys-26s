# 任务 1.2 — `bool isContiguous() const`

## 任务本质

张量在经过 `permute`（维度重排）、`slice`（切片）等操作后，逻辑上的数据顺序和物理内存中的存储顺序可能不再一致。`isContiguous()` 的本质是**检验张量的 strides（步长）是否恰好等于"按行优先（C-order）紧密排列"时应有的步长**。如果是，说明遍历张量元素的顺序正好对应内存中连续的字节，这对后续的 `view`、`reshape` 等操作是否合法至关重要——只有连续的张量才能直接重新解释形状。

## 目标

检查张量的形状（shape）和步长（strides），判断它在内存中是否连续存储。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/tensor/tensor.cpp` | 需要修改的实现文件，`isContiguous()` 在第 166-169 行 |
| `src/tensor/tensor.hpp` | `TensorMeta` 结构体定义了 `shape` 和 `strides` |

### 关键数据结构

```cpp
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;        // 各维度大小
    std::vector<ptrdiff_t> strides;   // 各维度步长（以元素为单位）
};
```

### 参考模式 — `create()` 中的步长计算（tensor.cpp:18-24）

`create()` 生成的是标准的行优先连续步长：

```cpp
size_t stride = 1;
for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;      // 最后一维步长=1
    stride *= shape[ndim_ - i];       // 依次向前累乘
}
```

对于 shape = [2, 3, 4]，生成的连续步长为 [12, 4, 1]：
- `strides[2] = 1`
- `strides[1] = 1 × 4 = 4`
- `strides[0] = 4 × 3 = 12`

`isContiguous()` 要做的就是**验证当前的 strides 是否符合这个规则**。

## 实现计划

### 修改位置

- **文件**: `src/tensor/tensor.cpp`
- **行**: 166-169

### 实现代码

```cpp
bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1;
    for (size_t i = _meta.shape.size(); i > 0; i--) {
        if (_meta.strides[i - 1] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i - 1]);
    }
    return true;
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `ptrdiff_t expected_stride = 1;` | 连续张量的最后一维步长必定为 1 |
| 2 | `for (size_t i = shape.size(); i > 0; i--)` | 从最后一维向第一维反向遍历 |
| 3 | `if (strides[i-1] != expected_stride) return false;` | 若实际步长 ≠ 期望步长，则不连续 |
| 4 | `expected_stride *= shape[i-1];` | 期望步长 = 当前维度大小 × 之前的累积步长 |
| 5 | `return true;` | 全部维度都匹配，则张量连续 |

### 示例验证

| shape | strides | 连续？ | 原因 |
|-------|---------|--------|------|
| [2, 3, 4] | [12, 4, 1] | ✅ | 标准行优先 |
| [2, 3, 4] | [4, 12, 1] | ❌ | 经过 permute 后维度互换 |
| [2, 3] | [3, 1] | ✅ | 标准行优先 |
| [3] | [2] | ❌ | slice 后步长不为 1 |

## 验证方式

- 编译项目确认无语法错误
- 运行项目测试用例
