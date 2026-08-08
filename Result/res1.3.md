# 结果记录 1.3 — `tensor_t view(const std::vector<size_t> &shape) const`

## 修改文件

`src/tensor/tensor.cpp` — 第 182-185 行

## 代码变化

### 修改前

```cpp
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
```

### 修改后

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

### Diff

```diff
 tensor_t Tensor::view(const std::vector<size_t> &shape) const {
-    TO_BE_IMPLEMENTED();
-    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
+    // 1. 张量必须内存连续
+    CHECK_ARGUMENT(isContiguous(), "view() requires a contiguous tensor");
+
+    // 2. 新旧 shape 的元素总数必须一致
+    size_t new_numel = 1;
+    for (size_t s : shape) new_numel *= s;
+    CHECK_ARGUMENT(new_numel == numel(), "view() shape is incompatible: element count mismatch");
+
+    // 3. 计算新的行优先连续 strides
+    size_t ndim_ = shape.size();
+    std::vector<ptrdiff_t> new_strides(ndim_);
+    size_t stride = 1;
+    for (size_t i = 1; i <= ndim_; i++) {
+        new_strides[ndim_ - i] = stride;
+        stride *= shape[ndim_ - i];
+    }
+
+    // 4. 构建新 TensorMeta，共享同一 _storage
+    TensorMeta new_meta{_meta.dtype, shape, new_strides};
+    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
 }
```

## 逐行解释

| 步骤 | 代码 | 为什么 |
|------|------|--------|
| 1 | `CHECK_ARGUMENT(isContiguous(), ...)` | `view` 不复制数据，只重解释内存布局。若张量经过 `permute` / `slice` 后 strides 不规整，新的行优先步长无法正确寻址原数据，必须拒绝操作。 |
| 2 | `new_numel == numel()` | 新旧 shape 映射到同一块内存，元素总数必须完全相同，否则会越界或浪费内存。 |
| 3 | strides 计算循环 | 为新 shape 生成标准行优先步长，与 `create()` 第 20-24 行逻辑完全一致（从最后一维向前累积乘积）。 |
| 4 | `new Tensor(new_meta, _storage, _offset)` | 共享同一 `_storage` 指针（无数据拷贝），保留原始 `_offset`（支持切片后再 view），仅替换 shape 和 strides 元数据。 |

## 设计模式参考

`view` 中步长计算与 `create()` 完全相同，只是作用对象不同：

```cpp
// create() 中：为全新张量生成步长
size_t stride = 1;
for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;
    stride *= shape[ndim_ - i];
}

// view() 中：为新 shape 重新生成步长（逻辑相同）
size_t stride = 1;
for (size_t i = 1; i <= ndim_; i++) {
    new_strides[ndim_ - i] = stride;
    stride *= shape[ndim_ - i];
}
```

与 task 1.1 的 `load()` 不同，`view()` 构造新张量时传入了 `_offset`，这确保了对切片张量再执行 view 时偏移量被正确继承。
