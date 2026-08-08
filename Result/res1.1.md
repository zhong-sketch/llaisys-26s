# 结果记录 1.1 — `void Tensor::load(const void *src)`

## 修改文件

`src/tensor/tensor.cpp` — 第 186-188 行

## 代码变化

### 修改前

```cpp
void Tensor::load(const void *src_) {
    TO_BE_IMPLEMENTED();
}
```

### 修改后

```cpp
void Tensor::load(const void *src_) {
    size_t bytes = numel() * elementSize();
    core::context().setDevice(this->deviceType(), this->deviceId());
    llaisysMemcpyKind_t kind = _storage->isHost() ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, kind);
}
```

### Diff

```diff
 void Tensor::load(const void *src_) {
-    TO_BE_IMPLEMENTED();
+    size_t bytes = numel() * elementSize();
+    core::context().setDevice(this->deviceType(), this->deviceId());
+    llaisysMemcpyKind_t kind = _storage->isHost() ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
+    core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, kind);
 }
```

## 逐行解释

| 行 | 代码 | 为什么 |
|----|------|--------|
| 1 | `size_t bytes = numel() * elementSize();` | 计算需要拷贝的总字节数。`numel()` 返回张量中所有元素的数量，`elementSize()` 返回每个元素的字节大小（如 float 为 4 字节）。两者相乘即为整个张量数据占用的内存大小。 |
| 2 | `core::context().setDevice(this->deviceType(), this->deviceId());` | 激活张量所在设备的运行时上下文。`context()` 获取线程局部的 `Context` 对象，`setDevice()` 切换到张量对应的设备（CPU 或 GPU）。这一步保证后续调用的运行时 API 是匹配该设备的。参考自 `debug()` 方法（第 150 行）和 `create()` 方法（第 33 行）中的相同模式。 |
| 3 | `llaisysMemcpyKind_t kind = _storage->isHost() ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;` | 根据张量存储位置选择正确的内存拷贝方向。`src` 参数始终是主机（CPU）指针。如果张量本身也在 CPU 上（`isHost() == true`），则是 Host→Host 拷贝；如果张量在 GPU 上，则是 Host→Device 拷贝。 |
| 4 | `core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, kind);` | 调用运行时 API 的同步内存拷贝函数。`this->data()` 是张量存储的目标地址（含偏移），`src_` 是源主机指针，`bytes` 是拷贝大小，`kind` 是方向。使用 `memcpy_sync` 而非 `memcpy_async` 确保函数返回时数据已完全拷贝完成。 |

## 设计模式参考

本实现直接参考了同文件中 `debug()` 方法（第 149-164 行）的 D2H 拷贝模式，将其反转为 H2D/H2H 方向：

```cpp
// debug() 中的 D2H 拷贝（参考）
core::context().setDevice(this->deviceType(), this->deviceId());
core::context().runtime().api()->memcpy_sync(
    tmp_tensor->data(), this->data(),
    this->numel() * this->elementSize(),
    LLAISYS_MEMCPY_D2H);
```
