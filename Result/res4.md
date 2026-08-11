# 结果记录 4 - 在 LLAISYS 中集成 CUDA

## 修改/新建文件

| 文件 | 操作 | 作用 |
|---|---|---|
| `plans/task4.1-nvidia.md` | 新建/更新 | NVIDIA CUDA 算力接入计划，并明确与天数后端隔离 |
| `plans/task4.2-muxi.md` | 新建/保留 | 沐曦后端预案 |
| `plans/task4.3-iluvatar.md` | 新建/更新 | 天数智芯后端预案 |
| `xmake.lua` | 修改 | `--nv-gpu=y` 时接入 NVIDIA device/ops target |
| `xmake/nvidia.lua` | 新建 | 定义 CUDA Runtime 和 CUDA 算子的 Xmake target |
| `src/device/nvidia/nvidia_runtime_api.cu` | 修改 | 实现 NVIDIA CUDA Runtime API |
| `src/device/nvidia/nvidia_resource.cu` | 修改 | 补齐 `Resource` 析构定义 |
| `src/ops/nvidia/common.cuh` | 新建 | CUDA dtype 转换、错误检查、launch 配置 |
| `src/ops/add/nvidia/add_nvidia.*` | 新建 | NVIDIA `add` 算子 |
| `src/ops/argmax/nvidia/argmax_nvidia.*` | 新建 | NVIDIA `argmax` 算子 |
| `src/ops/embedding/nvidia/embedding_nvidia.*` | 新建 | NVIDIA `embedding` 算子 |
| `src/ops/linear/nvidia/linear_nvidia.*` | 新建 | NVIDIA `linear` 算子 |
| `src/ops/rms_norm/nvidia/rms_norm_nvidia.*` | 新建 | NVIDIA `rms_norm` 算子 |
| `src/ops/rope/nvidia/rope_nvidia.*` | 新建 | NVIDIA `rope` 算子 |
| `src/ops/self_attention/nvidia/self_attention_nvidia.*` | 新建 | NVIDIA `self_attention` 算子 |
| `src/ops/swiglu/nvidia/swiglu_nvidia.*` | 新建 | NVIDIA `swiglu` 算子 |
| `src/ops/*/op.cpp` | 修改 | NVIDIA 分支从占位实现改为调用 CUDA 实现 |

## 代码变化

### 1. Xmake CUDA 编译配置

#### 修改前

仓库只有 CPU 配置：

```lua
includes("xmake/cpu.lua")
```

`--nv-gpu=y` 会引用 `xmake/nvidia.lua`，但该文件不存在。

#### 修改后

新增：

```lua
target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    add_rules("cuda")
    add_files("../src/device/nvidia/*.cu")
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    add_files("../src/ops/*/nvidia/*.cu")
target_end()
```

并在 `xmake.lua` 中接入：

```diff
 target("llaisys-device")
     add_deps("llaisys-device-cpu")
+    if has_config("nv-gpu") then
+        add_deps("llaisys-device-nvidia")
+    end

 target("llaisys-ops")
     add_deps("llaisys-ops-cpu")
+    if has_config("nv-gpu") then
+        add_deps("llaisys-ops-nvidia")
+    end
```

### 2. CUDA Runtime API

#### 修改前

```cpp
int getDeviceCount() {
    TO_BE_IMPLEMENTED();
}

void *mallocDevice(size_t size) {
    TO_BE_IMPLEMENTED();
}

void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    TO_BE_IMPLEMENTED();
}
```

#### 修改后

```cpp
int getDeviceCount() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    checkCuda(error, "cudaGetDeviceCount");
    return count;
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)),
              "cudaMemcpy");
}
```

同时修复了 `memcpyAsync` 的函数签名，使它和 `include/llaisys/runtime.h` 中的函数指针一致：

```cpp
void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream)
```

### 3. NVIDIA 算子调度

#### 修改前

各算子的 NVIDIA 分支仍是占位：

```cpp
#ifdef ENABLE_NVIDIA_API
case LLAISYS_DEVICE_NVIDIA:
    TO_BE_IMPLEMENTED();
    return;
#endif
```

#### 修改后

以 `add` 为例：

```cpp
#ifdef ENABLE_NVIDIA_API
case LLAISYS_DEVICE_NVIDIA:
    return nvidia::add(c->data(), a->data(), b->data(),
                       c->dtype(), c->numel());
#endif
```

其余 `argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu` 也采用同样模式：上层 `op.cpp` 只做检查和设备分发，具体 CUDA kernel 放在对应 `nvidia/` 子目录。

### 4. CUDA kernel 设计

| 算子 | 实现逻辑 |
|---|---|
| `add` | 一个线程处理一个元素，执行 `a[i] + b[i]` |
| `swiglu` | 一个线程处理一个元素，执行 `up * gate * sigmoid(gate)` |
| `embedding` | 一个线程处理一个输出字节，根据 token id 复制对应 embedding 行 |
| `linear` | 一个线程处理一个输出元素 `[i, j]`，内部循环 `K` 做点积 |
| `rms_norm` | 一个线程处理一行，先求平方均值，再归一化并乘权重 |
| `rope` | 一个线程处理一个旋转 pair，计算 sin/cos 后写回两个半维度 |
| `self_attention` | 一个线程处理一个输出元素，内部完成 QK 点积、causal mask、softmax、V 加权 |
| `argmax` | 一个 kernel 单线程扫描，输出最大值和最大索引 |

这是一版以正确性和教学清晰度为主的 CUDA 实现，不是最终高性能版本。后续性能优化可以把 `linear` 替换成 cuBLAS，把 attention 换成并行归约或 fused attention。

## 逐行逻辑解释

### Runtime 关键逻辑

| 代码 | 含义 |
|---|---|
| `checkCuda(error, "...")` | 将 CUDA 错误码转换成 C++ 异常，便于上层发现错误 |
| `toCudaMemcpyKind(kind)` | 把 LLAISYS 的 H2D/D2H/D2D 枚举转换为 CUDA 的 memcpy 方向 |
| `cudaGetDeviceCount(&count)` | 查询 NVIDIA GPU 数量 |
| `cudaSetDevice(device_id)` | 激活指定 GPU |
| `cudaStreamCreate(&stream)` | 创建 CUDA stream |
| `cudaMalloc(&ptr, size)` | 在 GPU 显存中分配 Tensor storage |
| `cudaMallocHost(&ptr, size)` | 分配 pinned host memory |
| `cudaMemcpy(...)` | 同步拷贝 CPU/GPU 内存 |
| `cudaMemcpyAsync(..., stream)` | 在指定 stream 上异步拷贝 |

### 算子关键逻辑

| 代码 | 含义 |
|---|---|
| `blocksFor(numel)` | 根据元素数计算 CUDA grid 大小 |
| `blockIdx.x * blockDim.x + threadIdx.x` | 计算当前线程负责的线性索引 |
| `loadAsFloat(...)` | 将 f32/f16/bf16 统一转成 float 计算 |
| `storeFromFloat<T>(...)` | 将 float 结果转回原始 dtype |
| `checkKernel("...")` | 检查 kernel launch 是否成功 |

## 验证结果

### 作业 #3 补充确认

用户要求先确认作业 #3 是否做过 `tensorDebug` 级别的数据核对。实际情况：

1. 作业 #3 原验证方式是端到端 token 对齐，并没有打印 Qwen2 每层中间张量。
2. 本次补充执行了一个小张量 `tensor.debug()` 对比 PyTorch 的检查，数据完全一致。
3. 重新执行了 Qwen2 最短推理测试，LLAISYS 与 HuggingFace token 完全一致。

执行命令：

```powershell
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'
D:\LLAISYS\env\python\.venv\Scripts\python.exe test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1
```

结果：

```text
Test passed!
```

### CPU 回归

```powershell
git diff --check
D:\LLAISYS\env\xmake\xmake.exe
D:\LLAISYS\env\xmake\xmake.exe install
D:\LLAISYS\env\python\.venv\Scripts\python.exe test/test_runtime.py --device cpu
D:\LLAISYS\env\python\.venv\Scripts\python.exe test/ops/add.py --device cpu
```

结果：

```text
build ok
install ok
test_runtime.py --device cpu: Test passed!
test/ops/add.py --device cpu: Test passed!
```

### CUDA 配置验证

尝试执行：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=y -cv
```

本机结果：

```text
checking for Cuda SDK directory ... no
error: Cuda SDK not found!
```

说明当前机器没有可被 Xmake 检测到的 CUDA SDK，因此无法在本机完成 NVIDIA CUDA 编译和 `--device nvidia` 运行测试。已将配置切回：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=n -cv
```

保证项目仍可正常按 CPU 模式构建和运行。

## 第二平台说明

作业截图要求从 NVIDIA、天数、摩尔、沐曦中至少选择两种平台。当前仓库已有且测试工具支持的设备枚举只有：

```cpp
LLAISYS_DEVICE_CPU
LLAISYS_DEVICE_NVIDIA
```

Python 测试参数也只有：

```text
choices=["cpu", "nvidia"]
```

因此本次没有伪造第二个类 CUDA 平台的完成状态。若要继续支持天数、摩尔线程或沐曦，需要先补：

1. `include/llaisys.h` 的设备枚举。
2. `python/llaisys/libllaisys/llaisys_types.py` 的设备映射。
3. 对应 `src/device/<vendor>/` Runtime。
4. 对应 `xmake/<vendor>.lua`。
5. 对应 vendor SDK 的编译器、头文件、库路径。
6. `test_utils.py` 和测试脚本中的设备名称。

## 结论

作业 #4 的 NVIDIA CUDA 代码路径已经按计划实现：

1. Runtime API 已由 CUDA Runtime API 接管。
2. Xmake 已新增 NVIDIA CUDA 编译 target。
3. Qwen2 所需的 8 个基础算子已新增 NVIDIA kernel。
4. 算子调度层已接入 `LLAISYS_DEVICE_NVIDIA` 分支。
5. CPU 路径回归测试通过。

当前唯一阻塞是本机缺少 CUDA SDK，无法本地执行 `xmake f --nv-gpu=y` 后的 CUDA 编译和 `--device nvidia` 测试。

## 与天数后端的兼容性补充

后续执行天数智芯计划时，不应复用 NVIDIA 的设备枚举、宏、Xmake 开关或源码目录。两个后端按并列关系存在：

```text
NVIDIA:
  LLAISYS_DEVICE_NVIDIA = 1
  ENABLE_NVIDIA_API
  --nv-gpu
  src/device/nvidia/
  src/ops/*/nvidia/

ILUVATAR:
  LLAISYS_DEVICE_ILUVATAR = 2
  ENABLE_ILUVATAR_API
  --iluvatar-gpu
  src/device/iluvatar/
  src/ops/*/iluvatar/
```

推荐在不同硬件环境中分别编译：

```bash
# NVIDIA 环境
xmake f --nv-gpu=y --iluvatar-gpu=n -cv

# 天数智芯环境
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
```

这样可以避免 CUDA SDK 和天数 SDK 的编译器、头文件、运行时库路径互相干扰。

### 2026-08-12 补充更新

根据后续需要同时保留 NVIDIA 与天数智芯两条实施路线，已将原先较宽泛的作业 #4 计划拆分为三个平台计划：

```text
plans/task4.1-nvidia.md
plans/task4.2-muxi.md
plans/task4.3-iluvatar.md
```

NVIDIA 计划的执行结果仍记录在本文件中；天数智芯计划的执行结果另见：

```text
Result/res4.3-iluvatar.md
```

本次修改没有改变 NVIDIA 源码目录、设备枚举值、编译宏或 Xmake 开关，只是在文档中明确要求 NVIDIA 环境使用：

```bash
xmake f --nv-gpu=y --iluvatar-gpu=n -cv
```

这样后续执行天数智芯计划时不会和 NVIDIA CUDA SDK、头文件、库路径发生混用。
