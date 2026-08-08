# 结果记录 1.2 — `bool isContiguous() const`

## 修改文件

`src/tensor/tensor.cpp` — 第 166-169 行

## 代码变化

### 修改前

```cpp
bool Tensor::isContiguous() const {
    TO_BE_IMPLEMENTED();
    return true;
}
```

### 修改后

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

### Diff

```diff
 bool Tensor::isContiguous() const {
-    TO_BE_IMPLEMENTED();
+    ptrdiff_t expected_stride = 1;
+    for (size_t i = _meta.shape.size(); i > 0; i--) {
+        if (_meta.strides[i - 1] != expected_stride) {
+            return false;
+        }
+        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i - 1]);
+    }
     return true;
 }
```

## 逐行解释

| 行 | 代码 | 为什么 |
|----|------|--------|
| 1 | `ptrdiff_t expected_stride = 1;` | 行优先（C-order）连续张量的**最后一维**步长固定为 1，从此处开始向前累积计算期望步长。 |
| 2 | `for (size_t i = _meta.shape.size(); i > 0; i--)` | 从最后一维（索引 `size-1`）向第一维（索引 `0`）反向遍历，与 `create()` 中生成步长的方向相同。 |
| 3 | `if (_meta.strides[i - 1] != expected_stride) return false;` | 将当前维度的实际步长与期望步长比较。若不一致，说明张量经过了 `permute` 或 `slice` 等操作，内存不连续，立即返回 false。 |
| 4 | `expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i - 1]);` | 更新下一维的期望步长：当前期望步长 × 当前维度大小。例如对 shape=[2,3,4]：第3维期望=1，第2维期望=1×4=4，第1维期望=4×3=12。 |
| 5 | `return true;` | 所有维度的步长均符合连续排列规则，张量在内存中连续。 |

## 设计模式参考

本实现与 `create()` 中的步长生成逻辑（第 20-24 行）互为镜像：

```cpp
// create() 中：生成连续步长
size_t stride = 1;
for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;      // 写入步长
    stride *= shape[ndim_ - i];       // 累积
}

// isContiguous() 中：验证步长是否连续
ptrdiff_t expected_stride = 1;
for (size_t i = shape.size(); i > 0; i--) {
    if (strides[i-1] != expected_stride) return false;  // 对比步长
    expected_stride *= shape[i - 1];                    // 累积
}
```

两者遍历方向、累积逻辑完全相同，区别仅在于一个是**写入**步长，另一个是**验证**步长。
