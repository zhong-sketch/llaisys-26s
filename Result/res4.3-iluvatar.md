# 结果记录 4.3 - 天数智芯 ILUVATAR 后端接入

## 任务本质

天数智芯计划的目标是让 LLAISYS 增加一个独立的 `iluvatar` 设备后端，使 Python 前端、C API、C++ 调度层、Runtime API 和算子入口都能通过统一接口访问天数算力。

## 冲突检查

NVIDIA 与天数后端可以同时保留在仓库中，不冲突：

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

推荐按平台分别构建：

```bash
# NVIDIA
xmake f --nv-gpu=y --iluvatar-gpu=n -cv

# 天数智芯
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
```

## 修改/新建文件

| 文件 | 作用 |
|---|---|
| `include/llaisys.h` | 新增 `LLAISYS_DEVICE_ILUVATAR = 2` |
| `python/llaisys/libllaisys/llaisys_types.py` | 新增 Python 侧 `DeviceType.ILUVATAR` |
| `python/llaisys/models/qwen2.py` | 支持字符串设备名 `"iluvatar"` |
| `test/test_utils.py` | 支持测试脚本映射 `"iluvatar"` |
| `test/test_runtime.py`、`test/test_infer.py`、`test/ops/*.py` | argparse 设备参数增加 `"iluvatar"` |
| `src/device/runtime_api.hpp` | 声明 `iluvatar::getRuntimeAPI()` |
| `src/device/runtime_api.cpp` | 新增 `LLAISYS_DEVICE_ILUVATAR` Runtime 分发 |
| `xmake.lua` | 新增 `--iluvatar-gpu` 编译开关 |
| `xmake/iluvatar.lua` | 定义天数 device/ops target |
| `src/device/iluvatar/` | 天数 CUDA 兼容 Runtime |
| `src/ops/*/iluvatar/` | 8 个天数算子入口和 fallback 实现 |
| `src/ops/*/op.cpp` | 新增 `LLAISYS_DEVICE_ILUVATAR` 算子分发 |

## 实现说明

### Runtime

天数 Runtime 使用 CUDA 兼容 API 接通：

```text
cudaGetDeviceCount
cudaSetDevice
cudaMalloc / cudaFree
cudaMallocHost / cudaFreeHost
cudaMemcpy / cudaMemcpyAsync
cudaStreamCreate / cudaStreamSynchronize / cudaStreamDestroy
```

### 算子

已覆盖 8 个作业算子：

```text
add
argmax
embedding
linear
rms_norm
rope
self_attention
swiglu
```

当前实现为正确性优先 fallback：

```text
Iluvatar device pointer
  -> cudaMemcpy DeviceToHost
  -> llaisys::ops::cpu::<op>()
  -> cudaMemcpy HostToDevice
```

这样可以验证设备分发、显存拷贝、统一 Runtime API 和结果正确性。后续性能优化可以将 fallback 替换为天数原生 GPU kernel。

## 实机验证

环境：

```text
Platform: Gitee compute instance
GPU: Iluvatar BI-V150
GPU memory: 32768 MiB
Runtime: CoreX / CUDA-compatible runtime
Python: 3.12.3
PyTorch: 2.7.1
torch.cuda.is_available(): True
```

构建：

```bash
source ~/.xmake/profile
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

xmake f --nv-gpu=n --iluvatar-gpu=y -cv
xmake
xmake install
```

测试：

```bash
python test/test_runtime.py --device iluvatar
python test/ops/add.py --device iluvatar
python test/ops/argmax.py --device iluvatar
python test/ops/embedding.py --device iluvatar
python test/ops/linear.py --device iluvatar
python test/ops/rms_norm.py --device iluvatar
python test/ops/rope.py --device iluvatar
python test/ops/self_attention.py --device iluvatar
python test/ops/swiglu.py --device iluvatar
```

结果：

```text
Found 1 iluvatar devices
runtime test: Test passed!
all 8 iluvatar op tests: Test passed!
```

## 结论

天数智芯后端已完成独立设备接入，并在 Gitee BI-V150 实机上通过 Runtime 和 8 个算子测试。当前版本的算子实现强调正确性和链路连通性，不代表最终高性能实现；这一点已在提交报告中明确说明。
