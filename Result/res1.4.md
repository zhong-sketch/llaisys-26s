# 结果记录 1.4 — `tensor_t permute(const std::vector<size_t> &order) const`

## 修改文件

`src/tensor/tensor.cpp` — 第 177-180 行

## 代码变化

### 修改前

```cpp
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
```

### 修改后

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

### Diff

```diff
 tensor_t Tensor::permute(const std::vector<size_t> &order) const {
-    TO_BE_IMPLEMENTED();
-    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
+    size_t ndim_ = _meta.shape.size();
+
+    // 1. 校验 order 长度
+    CHECK_ARGUMENT(order.size() == ndim_,
+                   "permute() order size must match tensor ndim");
+
+    // 2. 校验 order 是合法排列（无越界、无重复）
+    std::vector<bool> seen(ndim_, false);
+    for (size_t idx : order) {
+        CHECK_ARGUMENT(idx < ndim_, "permute() order index out of range");
+        CHECK_ARGUMENT(!seen[idx], "permute() order contains duplicate index");
+        seen[idx] = true;
+    }
+
+    // 3. 按 order 重排 shape 和 strides
+    std::vector<size_t> new_shape(ndim_);
+    std::vector<ptrdiff_t> new_strides(ndim_);
+    for (size_t i = 0; i < ndim_; i++) {
+        new_shape[i] = _meta.shape[order[i]];
+        new_strides[i] = _meta.strides[order[i]];
+    }
+
+    // 4. 构建新 TensorMeta，共享同一 _storage
+    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
+    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
 }
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `order.size() == ndim_` | `order` 必须为每一个维度指定一个新位置，长度不匹配则无法建立完整的映射 |
| 2 | `seen` 数组 + 双重校验 | 首先检查索引不越界（`idx < ndim_`），再通过 `seen` 标记确保每个维度下标恰好出现一次（合法排列）；若重复则访问到同一维两次、遗漏另一维 |
| 3 | `new_shape[i] = shape[order[i]]` | 新张量的第 i 维对应原张量的第 `order[i]` 维，shape 和 strides **同步**按相同下标重排，保证步长与形状一致 |
| 4 | `new Tensor(new_meta, _storage, _offset)` | 共享 `_storage`（零拷贝），保留 `_offset`，只更换元数据——这正是 permute 不移动数据的关键所在 |

## 设计模式参考

与 task 1.3 `view()` 的构造模式完全一致，区别在于生成新 shape/strides 的方式：

```cpp
// view()：根据新 shape 重新计算行优先 strides（结果一定连续）
new_strides[ndim_ - i] = stride;
stride *= shape[ndim_ - i];

// permute()：按 order 重新索引原有 shape/strides（结果通常不连续）
new_shape[i]   = _meta.shape[order[i]];
new_strides[i] = _meta.strides[order[i]];
```

两者都共享 `_storage` 和 `_offset`，都是纯元数据操作。
