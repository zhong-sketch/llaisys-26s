# 结果记录 4 - CUDA/类 CUDA 平台接入

## 任务本质

作业 #4 的目标是在 LLAISYS 中接入 CUDA 或类 CUDA 平台，让 Python 前端可以通过统一设备接口调用后端 Runtime、设备内存、算子和 Qwen2 推理路径。

本次完成并验证的平台：

| 平台 | 完成状态 |
|---|---|
| NVIDIA | Runtime、CUDA 编译、8 个 GPU 算子、Qwen2 GPU 推理均已实机通过 |
| 天数智芯 ILUVATAR | Runtime 和 8 个算子已在 Gitee BI-V150 实机通过，算子为正确性优先 fallback |
| CPU | 作为基线完成回归测试 |
| 沐曦 | 仅保留计划，算力不可用，未执行 |

## 修改/新建文件

| 文件 | 作用 |
|---|---|
| `plans/task4.1-nvidia.md` | NVIDIA CUDA 算力接入计划和验证记录 |
| `plans/task4.2-muxi.md` | 沐曦平台预案 |
| `plans/task4.3-iluvatar.md` | 天数智芯平台计划 |
| `xmake.lua` | 增加 NVIDIA/ILUVATAR 后端编译开关 |
| `xmake/nvidia.lua` | NVIDIA CUDA 编译 target 和 `cudart` 链接配置 |
| `xmake/iluvatar.lua` | 天数智芯后端编译 target |
| `src/device/nvidia/` | NVIDIA CUDA Runtime API |
| `src/device/iluvatar/` | 天数智芯 CUDA 兼容 Runtime API |
| `src/ops/*/nvidia/` | NVIDIA 8 个 GPU 算子 |
| `src/ops/*/iluvatar/` | 天数智芯 8 个 fallback 算子 |
| `src/ops/nvidia/common.cuh` | CUDA dtype 转换、kernel launch 和错误检查 |
| `python/llaisys/libllaisys/__init__.py` | Windows 下自动加入 CUDA DLL 搜索路径 |
| `src/core/context/context.cpp` | Runtime 延迟初始化，避免导入阶段初始化 CUDA |
| `src/core/runtime/runtime.cpp` | Runtime 析构同步和 stream 清理 |
| `test/ops/self_attention.py` | 修复 NVIDIA 测试中 mask 的设备位置 |
| `submission_reports/task4/README.md` | 作业 #4 简要复现报告 |
| `submission_reports/task4/pr_description.md` | 作业 #4 PR 描述草稿 |

## NVIDIA 验证

环境：

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Toolkit: 12.8
nvcc: 12.8.93
PyTorch: 2.11.0+cu128
Python venv: D:\LLAISYS\env\python\.venv-nvidia
```

构建：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=y --iluvatar-gpu=n -cv
D:\LLAISYS\env\xmake\xmake.exe -y
D:\LLAISYS\env\xmake\xmake.exe install -y
```

测试通过：

```text
test/test_runtime.py --device nvidia
test/ops/add.py --device nvidia
test/ops/argmax.py --device nvidia
test/ops/embedding.py --device nvidia
test/ops/linear.py --device nvidia
test/ops/rms_norm.py --device nvidia
test/ops/rope.py --device nvidia
test/ops/self_attention.py --device nvidia
test/ops/swiglu.py --device nvidia
test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device nvidia
```

推理结果：

```text
LLAISYS tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]

HuggingFace/PyTorch tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]

Test passed!
exit code: 0
```

## 天数智芯验证

环境：

```text
Platform: Gitee compute instance
GPU: Iluvatar BI-V150
GPU memory: 32768 MiB
Runtime: CoreX / CUDA-compatible runtime
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

测试通过：

```text
test/test_runtime.py --device iluvatar
test/ops/add.py --device iluvatar
test/ops/argmax.py --device iluvatar
test/ops/embedding.py --device iluvatar
test/ops/linear.py --device iluvatar
test/ops/rms_norm.py --device iluvatar
test/ops/rope.py --device iluvatar
test/ops/self_attention.py --device iluvatar
test/ops/swiglu.py --device iluvatar
```

说明：天数算子当前采用 D2H/CPU/H2D fallback，重点验证统一设备接口、Runtime、设备内存拷贝和算子正确性链路。该路径不是最终高性能原生天数 kernel。

## CPU 回归

测试通过：

```text
test/test_runtime.py --device cpu
test/test_tensor.py
all 8 op tests with --device cpu
test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device cpu
```

CPU Qwen2 推理 token 与 HuggingFace/PyTorch 完全一致，退出码为 0。

## 关键修复

1. Python 加载共享库前自动加入 CUDA DLL 搜索路径，解决 Windows 下 `cudart64_12.dll` 查找问题。
2. Context 默认只创建 CPU Runtime，GPU Runtime 在 `setDevice()` 时延迟创建，避免 CPU 测试导入时初始化 CUDA。
3. Runtime 析构时同步并销毁 stream，解决 Python 进程退出阶段异常。
4. NVIDIA fp16 转换改为 IEEE 754 最近偶数舍入，修复大尺寸 fp16 算子误差。
5. self-attention 测试的 mask 跟随 `query.device` 创建，避免 CPU/GPU 张量混用。

## 结论

作业 #4 当前满足提交材料要求：至少覆盖 NVIDIA 和天数智芯两个平台，并提供了复现报告和逐平台状态说明。需要在 PR 页面继续确认 GitHub Actions 的 CPU CI 结果为绿色；GPU 测试由于 GitHub hosted runner 没有对应硬件，需要以本地 NVIDIA 和 Gitee 天数实机测试记录作为补充。
