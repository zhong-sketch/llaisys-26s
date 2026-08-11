# 任务 4.2-MUXI - 沐曦国产算力接入计划

## 任务本质

沐曦是作业 #4 候选的国产类 CUDA 平台。本计划目标是为 LLAISYS 新增 `muxi` 后端，使上层 Python、C API、模型和算子调度可以通过统一设备接口调用沐曦算力。

当前限制是：Gitee 沐曦算力暂时售罄，因此该计划暂不执行，等待资源恢复后再推进。

## 目标

1. 新增 LLAISYS 设备类型：

```cpp
LLAISYS_DEVICE_MUXI
```

2. 新增 Python 设备映射：

```python
DeviceType.MUXI
```

3. 新增编译宏：

```cpp
ENABLE_MUXI_API
```

4. 新增 Xmake 开关：

```bash
xmake f --muxi-gpu=y -cv
```

5. 新增沐曦 Runtime 和 8 个 Qwen2 必需算子。

## 涉及文件

| 文件 | 计划操作 |
|---|---|
| `include/llaisys.h` | 新增 `LLAISYS_DEVICE_MUXI` |
| `python/llaisys/libllaisys/llaisys_types.py` | 新增 `DeviceType.MUXI` |
| `test/test_utils.py` | 新增 `"muxi"` 到 LLAISYS 和 PyTorch/测试映射 |
| `src/device/runtime_api.hpp` | 声明 `muxi::getRuntimeAPI()` |
| `src/device/runtime_api.cpp` | 分发 `LLAISYS_DEVICE_MUXI` |
| `src/device/muxi/` | 新增沐曦 Runtime |
| `xmake.lua` | 新增 `muxi-gpu` option |
| `xmake/muxi.lua` | 新增沐曦编译 target |
| `src/ops/*/muxi/` | 新增沐曦算子 |
| `src/ops/*/op.cpp` | 新增 MUXI 分支 |

## 实施步骤

### 第一步：确认沐曦实例环境

进入 Gitee 沐曦实例后，先运行：

```bash
uname -a
python --version
which xmake
```

再检查沐曦工具链，具体命令以实例镜像为准，可先尝试：

```bash
which maca-smi
which mxcc
maca-smi
```

目标是确认：

1. 设备可见。
2. SDK 已安装。
3. 编译器和运行时库路径可用。

### 第二步：新增设备枚举和 Python 映射

计划修改：

```cpp
LLAISYS_DEVICE_CPU = 0,
LLAISYS_DEVICE_NVIDIA = 1,
LLAISYS_DEVICE_MUXI = 2,
LLAISYS_DEVICE_TYPE_COUNT
```

Python 侧新增：

```python
class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
    MUXI = 2
```

### 第三步：实现沐曦 Runtime

新增：

```text
src/device/muxi/muxi_runtime_api.*
src/device/muxi/muxi_resource.*
```

需要实现：

```text
getDeviceCount
setDevice
deviceSynchronize
createStream
destroyStream
streamSynchronize
mallocDevice
freeDevice
mallocHost
freeHost
memcpySync
memcpyAsync
```

### 第四步：实现沐曦算子

优先顺序：

```text
add
swiglu
embedding
argmax
rms_norm
rope
linear
self_attention
```

如果沐曦支持类 CUDA kernel 语法，优先迁移 `src/ops/*/nvidia/` 的朴素 kernel；如果不兼容，则按沐曦 SDK 重写。

### 第五步：验证

Runtime：

```bash
python test/test_runtime.py --device muxi
```

算子：

```bash
python test/ops/add.py --device muxi
python test/ops/argmax.py --device muxi
python test/ops/embedding.py --device muxi
python test/ops/linear.py --device muxi
python test/ops/rms_norm.py --device muxi
python test/ops/rope.py --device muxi
python test/ops/self_attention.py --device muxi
python test/ops/swiglu.py --device muxi
```

推理：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device muxi \
  --test \
  --max_steps 1
```

## 风险点

1. 当前 Gitee 沐曦算力售罄，无法马上验证。
2. 沐曦 SDK 命令和 CUDA API 不一定完全一致，需要以实例环境为准。
3. PyTorch 是否支持沐曦后端要现场确认；若不支持，需要用 CPU PyTorch 生成答案，再拷回 CPU 比较。
