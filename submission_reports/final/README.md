# LLAISYS 作业 #1-#4 总提交报告

## 总览

本报告汇总 LLAISYS 四组作业的实现内容、复现方式和平台状态，供最终 Pull Request 使用。

| 作业 | 内容 | 状态 |
|---|---|---|
| #1 | Tensor 基础能力 | 已通过 |
| #2 | 8 个 CPU 算子 | 已通过 |
| #3 | Qwen2 CPU 推理 | 已通过 |
| #4 | NVIDIA 与天数智芯平台接入 | 已通过主要验证，天数算子为 fallback |

## 作业 #1：Tensor

完成内容：

- `Tensor::load`
- `Tensor::isContiguous`
- `Tensor::view`
- `Tensor::permute`
- `Tensor::slice`

验证：

```bash
python test/test_tensor.py
```

结果：测试通过。

## 作业 #2：算子

完成 8 个 CPU 算子：

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

验证：

```bash
python test/ops/add.py --device cpu
python test/ops/argmax.py --device cpu
python test/ops/embedding.py --device cpu
python test/ops/linear.py --device cpu
python test/ops/rms_norm.py --device cpu
python test/ops/rope.py --device cpu
python test/ops/self_attention.py --device cpu
python test/ops/swiglu.py --device cpu
```

结果：全部通过，覆盖 f32、f16、bf16 和大小尺寸测试。

## 作业 #3：Qwen2 推理

完成内容：

- Python 侧读取 Qwen2 配置和 safetensors 权重。
- 通过 ctypes 调用 C API 创建、销毁、加载和推理模型。
- C++ 后端实现 Qwen2 Transformer forward。
- 支持 KV cache。
- 使用 LLAISYS 算子完成 embedding、RMSNorm、RoPE、self-attention、SwiGLU、linear、add、argmax。

验证：

```powershell
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'
D:\LLAISYS\env\python\.venv-nvidia\Scripts\python.exe test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device cpu
```

结果：

```text
LLAISYS tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]

HuggingFace/PyTorch tokens:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]

Test passed!
exit code: 0
```

## 作业 #4：CUDA/类 CUDA 平台

### NVIDIA

验证环境：

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Toolkit: 12.8
nvcc: 12.8.93
PyTorch: 2.11.0+cu128
```

完成内容：

- CUDA Runtime API。
- Xmake CUDA 编译配置。
- 8 个 NVIDIA GPU 算子。
- Qwen2 `--device nvidia` 推理。

验证：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=y --iluvatar-gpu=n -cv
D:\LLAISYS\env\xmake\xmake.exe -y
D:\LLAISYS\env\xmake\xmake.exe install -y

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

结果：Runtime、8 个算子、Qwen2 GPU 推理全部通过，退出码为 0。

### 天数智芯 ILUVATAR

验证环境：

```text
Platform: Gitee compute instance
GPU: Iluvatar BI-V150
GPU memory: 32768 MiB
Runtime: CoreX / CUDA-compatible runtime
```

完成内容：

- 独立设备类型 `LLAISYS_DEVICE_ILUVATAR`。
- 独立编译开关 `--iluvatar-gpu`。
- CUDA 兼容 Runtime。
- 8 个算子 fallback。

验证：

```bash
source ~/.xmake/profile
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

xmake f --nv-gpu=n --iluvatar-gpu=y -cv
xmake
xmake install

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

结果：Runtime 和 8 个算子通过。当前天数算子是 D2H/CPU/H2D fallback，验证正确性和平台接入链路，不是最终高性能 kernel。

## 逐平台状态

| 平台 | 支持状态 | 说明 |
|---|---|---|
| CPU | 已通过 | GitHub Actions 可覆盖 CPU 回归 |
| NVIDIA | 已通过 | 本机 RTX 4070 Laptop GPU 实机验证 |
| 天数智芯 ILUVATAR | 已通过 Runtime 和算子 | Gitee BI-V150 实机验证，算子为 fallback |
| 沐曦 | 未执行 | 算力不可用，仅保留计划 |

## CI 说明

GitHub Actions 当前只验证 CPU 路径，因为 hosted runner 没有 NVIDIA 或天数智芯硬件。GPU 结果由本机和 Gitee 实机测试补充。提交 PR 后需要确认 CPU CI 为绿色。
