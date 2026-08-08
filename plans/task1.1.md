# 任务 1.1 — `void Tensor::load(const void *src)`

## 目标

将主机 (CPU) 内存中的原始数据加载到张量的存储中。张量可能位于 CPU 或 GPU（设备）上。

## 背景分析

### 涉及文件

| 文件 | 作用 |
|------|------|
| `src/tensor/tensor.cpp` | 需要修改的实现文件，`load()` 方法在第 186-188 行 |
| `src/tensor/tensor.hpp` | 张量类声明，`load()` 签名在第 52 行 |
| `src/core/context/context.hpp` | 提供 `context()` 全局函数获取线程局部上下文 |
| `src/core/runtime/runtime.hpp` | Runtime 类，提供 `api()` 获取运行时 API |
| `src/core/storage/storage.hpp` | Storage 类，提供 `isHost()` / `deviceType()` / `deviceId()` |
| `include/llaisys/runtime.h` | `LlaisysRuntimeAPI` 结构体，包含 `memcpy_sync` 函数指针 |
| `include/llaisys.h` | `llaisysMemcpyKind_t` 枚举定义 |

### 关键 API

```c
// memcpy_sync 签名
typedef void (*memcpy_sync_api)(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind);

// 拷贝方向枚举
typedef enum {
    LLAISYS_MEMCPY_H2H = 0,  // Host → Host
    LLAISYS_MEMCPY_H2D = 1,  // Host → Device
    LLAISYS_MEMCPY_D2H = 2,  // Device → Host
    LLAISYS_MEMCPY_D2D = 3,  // Device → Device
} llaisysMemcpyKind_t;
```

### 参考模式 — `debug()` 方法 (tensor.cpp:149-164)

`debug()` 中已有从设备到主机的内存拷贝模式：

```cpp
core::context().setDevice(this->deviceType(), this->deviceId());
core::context().runtime().api()->memcpy_sync(
    tmp_tensor->data(),                    // dst (host)
    this->data(),                          // src (device)
    this->numel() * this->elementSize(),   // 字节数
    LLAISYS_MEMCPY_D2H);                   // 方向: Device → Host
```

`load()` 需要做反方向的操作：从 Host 拷贝到张量。

## 实现计划

### 修改位置

- **文件**: `src/tensor/tensor.cpp`
- **行**: 186-188
- **操作**: 替换 `TO_BE_IMPLEMENTED()` 为实际实现

### 实现代码

```cpp
void Tensor::load(const void *src_) {
    size_t bytes = numel() * elementSize();
    core::context().setDevice(this->deviceType(), this->deviceId());
    llaisysMemcpyKind_t kind = _storage->isHost() ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, kind);
}
```

### 逻辑拆解

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `numel() * elementSize()` | 计算需要拷贝的总字节数（元素数量 × 单元素字节大小） |
| 2 | `setDevice(deviceType(), deviceId())` | 激活张量所在设备的运行时上下文 |
| 3 | `isHost() ? H2H : H2D` | 根据张量存储位置决定拷贝方向 |
| 4 | `memcpy_sync(...)` | 执行同步内存拷贝 |

## 验证方式

- 编译项目确认无语法错误
- 运行项目现有测试用例验证功能正确性
