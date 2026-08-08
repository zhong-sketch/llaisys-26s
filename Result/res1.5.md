# 结果记录 1.5 — `tensor_t slice(size_t dim, size_t start, size_t end) const`

## 修改文件

`src/tensor/tensor.cpp` — 第 228-231 行

## 代码变化

### 修改前

```cpp
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
```

### 修改后

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

### Diff

```diff
 tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
-    TO_BE_IMPLEMENTED();
-    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
+    // 1. 校验 dim 不越界
+    CHECK_ARGUMENT(dim < _meta.shape.size(),
+                   "slice() dim out of range");
+
+    // 2. 校验 start/end 合法
+    CHECK_ARGUMENT(start < end,
+                   "slice() start must be less than end");
+    CHECK_ARGUMENT(end <= _meta.shape[dim],
+                   "slice() end out of range for this dimension");
+
+    // 3. 构建新 shape：只修改被切片的那个维度
+    std::vector<size_t> new_shape = _meta.shape;
+    new_shape[dim] = end - start;
+
+    // 4. 计算新的字节偏移：在原 _offset 基础上加上 start 对应的偏移
+    size_t new_offset = _offset +
+                        static_cast<size_t>(_meta.strides[dim]) * start * elementSize();
+
+    // 5. strides 不变，直接复用 _meta
+    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
+    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
 }
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `dim < _meta.shape.size()` | 维度下标必须在 `[0, ndim)` 范围内，越界则访问无效内存 |
| 2 | `start < end` | 切片范围必须是非空区间，`start == end` 没有意义 |
| 2 | `end <= _meta.shape[dim]` | 切片终点不能超出该维度的总大小，否则会越界 |
| 3 | `new_shape[dim] = end - start` | 仅缩小被切片的维度大小，其余维度的 shape 不变，保持多维结构 |
| 4 | `new_offset = _offset + strides[dim] * start * elementSize()` | `strides[dim]` 是元素粒度的步长（跨过 `start` 个元素需要 `strides[dim] * start` 个元素），乘以 `elementSize()` 换算成字节偏移，累加在原有 `_offset` 上（支持链式切片） |
| 5 | strides 不变 | 切片只改变有效范围，不改变元素之间的内存间距，strides 依然正确描述各维度的跨步 |

## 设计模式参考

与 `permute()` 和 `view()` 相同，都共享 `_storage` 实现零拷贝，区别在于 offset 的处理：

```cpp
// permute()：offset 不变，shape/strides 重排
return new Tensor(new_meta, _storage, _offset);

// view()：offset 不变，shape 变/strides 重算
return new Tensor(new_meta, _storage, _offset);

// slice()：offset 前移，shape 缩小，strides 不变
return new Tensor(new_meta, _storage, _offset + strides[dim] * start * elementSize());
```

`slice` 是三者中唯一真正修改 `offset` 的操作。
