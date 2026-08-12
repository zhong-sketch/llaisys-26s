# 作业 #4 提交简要报告

## 1. 目标

作业 #4 要求在 LLAISYS 中接入 CUDA 或类 CUDA 平台，完成 Runtime API、GPU 算子和 GPU 推理验证，并在提交时逐个平台说明状态。

本次最终覆盖的平台：

| 平台 | 状态 |
|---|---|
| NVIDIA | 已在本机 RTX 4070 Laptop GPU 上通过 Runtime、8 个 CUDA 算子和 Qwen2 GPU 推理测试 |
| 天数智芯 ILUVATAR | 已在 Gitee BI-V150 实例上通过 Runtime 和 8 个算子测试，算子为正确性优先 fallback |
| CPU | 作为回归基线，Runtime、Tensor、8 个算子和 Qwen2 推理通过 |
| 沐曦 | 仅保留计划，因算力不可用未执行 |

## 2. NVIDIA 复现流程

验证环境：

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Toolkit: 12.8
nvcc: 12.8.93
PyTorch: 2.11.0+cu128
Python: D:\LLAISYS\env\python\.venv-nvidia
Model: D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B
```

构建：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=y --iluvatar-gpu=n -cv
D:\LLAISYS\env\xmake\xmake.exe -y
D:\LLAISYS\env\xmake\xmake.exe install -y
```

测试：

```powershell
$py='D:\LLAISYS\env\python\.venv-nvidia\Scripts\python.exe'
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'

& $py test/test_runtime.py --device nvidia
& $py test/ops/add.py --device nvidia
& $py test/ops/argmax.py --device nvidia
& $py test/ops/embedding.py --device nvidia
& $py test/ops/linear.py --device nvidia
& $py test/ops/rms_norm.py --device nvidia
& $py test/ops/rope.py --device nvidia
& $py test/ops/self_attention.py --device nvidia
& $py test/ops/swiglu.py --device nvidia
& $py test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device nvidia
```

结果：

```text
Found 1 nvidia devices
all 8 nvidia op tests: Test passed!
Qwen2 nvidia inference: Test passed!
exit code: 0
```

推理输出 token 与 HuggingFace/PyTorch 对齐：

```text
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]
```

## 3. 天数智芯复现流程

验证环境：

```text
Platform: Gitee compute instance
GPU: Iluvatar BI-V150
GPU memory: 32768 MiB
Runtime: CoreX / CUDA-compatible runtime
Python: 3.12.3
PyTorch: 2.7.1
torch.cuda.is_available(): True
```

构建和环境：

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

说明：天数算子当前是正确性优先的 D2H/CPU/H2D fallback，用于验证 Python 前端、C API、C++ 调度、天数 Runtime、设备内存拷贝和结果对齐链路。它不是最终高性能原生天数 kernel。

## 4. CPU 回归

```powershell
$py='D:\LLAISYS\env\python\.venv-nvidia\Scripts\python.exe'
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'

& $py test/test_runtime.py --device cpu
& $py test/test_tensor.py
& $py test/ops/add.py --device cpu
& $py test/ops/argmax.py --device cpu
& $py test/ops/embedding.py --device cpu
& $py test/ops/linear.py --device cpu
& $py test/ops/rms_norm.py --device cpu
& $py test/ops/rope.py --device cpu
& $py test/ops/self_attention.py --device cpu
& $py test/ops/swiglu.py --device cpu
& $py test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device cpu
```

结果：全部通过，CPU Qwen2 推理与 HuggingFace/PyTorch token 一致，退出码为 0。

## 5. 实现说明

NVIDIA 后端：

- `xmake/nvidia.lua` 配置 CUDA 编译、`cudart` 链接和 Windows `/MD` 编译选项。
- `src/device/nvidia/` 实现 CUDA Runtime API。
- `src/ops/*/nvidia/` 实现 `add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`。
- `src/ops/nvidia/common.cuh` 统一 dtype 转换、kernel launch 配置和 CUDA 错误检查。

天数后端：

- 使用独立设备枚举 `LLAISYS_DEVICE_ILUVATAR`。
- 使用独立编译开关 `--iluvatar-gpu` 和宏 `ENABLE_ILUVATAR_API`。
- 使用 CUDA 兼容 Runtime 路径连接 Gitee BI-V150。
- 8 个算子当前采用 fallback 验证正确性链路。

通用修复：

- Windows Python 加载共享库前自动加入 CUDA DLL 搜索路径。
- Context 默认只初始化 CPU Runtime，GPU Runtime 在 `setDevice()` 时延迟创建。
- Runtime 析构时同步并释放 stream，避免 Python 进程退出阶段崩溃。
- NVIDIA fp16 转换改为符合 IEEE 754 的最近偶数舍入。
- self-attention 测试中的 mask 跟随 query 设备创建，避免 CPU/GPU 张量混用。

## 6. CI 说明

GitHub Actions 当前只覆盖 CPU 测试，因为 GitHub 托管 runner 没有 NVIDIA 或天数智芯 GPU。GPU 部分通过本机 NVIDIA 和 Gitee 天数实例做实机验证；PR 发起后仍需要确认 GitHub Actions 的 CPU CI 为绿色。
