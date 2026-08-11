# 作业 #4 提交简要报告

## 1. 提交目标

本次作业目标是在 LLAISYS 中完成 GPU 后端接入验证，并按作业要求通过 Pull Request 提交到：

```text
wooway777/llaisys-26s
```

当前代码已提交并推送到个人 fork：

```text
zhong-sketch/llaisys-26s
```

关键提交：

```text
a147d46 Add task4 GPU backend scaffolding
15c7f02 Enable Iluvatar CUDA-compatible runtime
d98fb56 Add Iluvatar op fallback implementations
```

## 2. 复现流程

### 2.1 获取代码

在 Gitee 天数智芯算力实例中执行：

```bash
cd /data
wget -O task4-fallback.tar.gz "https://github.com/zhong-sketch/llaisys-26s/archive/refs/heads/main.tar.gz?ts=$(date +%s)"
tar -xzf task4-fallback.tar.gz
mv llaisys-26s-main llaisys-26s-task4-fallback
cd /data/llaisys-26s-task4-fallback
```

### 2.2 配置天数运行环境

```bash
source ~/.xmake/profile
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python
```

天数环境检查结果：

```text
GPU: Iluvatar BI-V150
GPU memory: 32768 MiB
IX-ML: 4.4.0
Driver Version: 4.4.0
Python: 3.12.3
PyTorch: 2.7.1
torch.cuda.is_available(): True
```

确认到的 CUDA 兼容路径：

```text
/usr/local/corex-4.4.0/include/cuda_runtime.h
/usr/local/corex-4.4.0/lib64/libcudart.so
/usr/local/cuda-10.2/include/cuda_runtime.h
```

### 2.3 编译与安装

```bash
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
xmake
xmake install
```

复现结果：

```text
build ok
install ok
```

### 2.4 Runtime 测试

```bash
python test/test_runtime.py --device iluvatar
```

复现结果：

```text
Found 1 iluvatar devices
Testing device {i}...
Passed
Test passed!
```

### 2.5 算子测试

已在天数实例上执行：

```bash
python test/ops/add.py --device iluvatar
python test/ops/argmax.py --device iluvatar
python test/ops/embedding.py --device iluvatar
python test/ops/linear.py --device iluvatar
python test/ops/rms_norm.py --device iluvatar
python test/ops/rope.py --device iluvatar
python test/ops/self_attention.py --device iluvatar
python test/ops/swiglu.py --device iluvatar
```

复现结果：

```text
全部通过
```

其中 `add` 测试覆盖了：

```text
shape (2, 3), dtype f32/f16/bf16
shape (512, 4096), dtype f32/f16/bf16
```

## 3. 平台支持状态

| 平台 | 编译开关 | 状态 | 说明 |
|---|---|---|---|
| CPU | 默认 | 通过 | 本地 Windows + MSVC 下 `test_runtime.py --device cpu` 通过 |
| NVIDIA | `--nv-gpu=y` | 已实现代码路径，未实机验证 | 本地无 CUDA SDK，Gitee NVIDIA 算力暂时不可用 |
| 天数智芯 ILUVATAR | `--iluvatar-gpu=y` | Runtime 与 8 个算子测试通过 | 在 Gitee `Iluvatar BI-V150` 实例上通过 |
| 沐曦 | 独立计划保留 | 未执行 | 算力暂时不可用，仅保留计划 |

## 4. 实现说明

### 4.1 Runtime

新增 `LLAISYS_DEVICE_ILUVATAR = 2`，并通过 `ENABLE_ILUVATAR_API` 与 `--iluvatar-gpu` 独立控制天数后端。

天数 Runtime 使用 CUDA 兼容 API：

```text
cudaGetDeviceCount
cudaSetDevice
cudaMalloc / cudaFree
cudaMallocHost / cudaFreeHost
cudaMemcpy / cudaMemcpyAsync
cudaStreamCreate / cudaStreamSynchronize / cudaStreamDestroy
```

### 4.2 算子

当前 8 个天数算子采用正确性优先的 fallback 实现：

```text
Iluvatar device pointer
  -> cudaMemcpy DeviceToHost
  -> llaisys::ops::cpu::<op>()
  -> cudaMemcpy HostToDevice
```

该实现可以验证 Python 前端、C API、C++ 调度、天数 Runtime、设备内存拷贝和算子结果对齐链路。后续性能优化可以将 fallback 替换为天数/CoreX 原生 GPU kernel。

## 5. CI 状态

作业要求 Pull Request 的 CI 必须通过。当前本地与天数实例测试均已通过；提交 PR 后，需要在 GitHub Pull Request 页面确认 CI 结果为绿色通过。

建议 PR 提交后检查：

```text
Actions / Checks: all passed
```

## 6. PR 描述建议

可将以下内容复制到 Pull Request 描述中：

```markdown
## Summary

- Add NVIDIA CUDA backend scaffolding and operator implementations.
- Add Iluvatar device type, runtime dispatch, xmake option, and Python test support.
- Implement Iluvatar CUDA-compatible runtime using CoreX CUDA runtime paths.
- Add correctness-first Iluvatar operator fallback implementations for add, argmax, embedding, linear, rms_norm, rope, self_attention, and swiglu.

## Verification

- CPU runtime test passed locally.
- Gitee Iluvatar BI-V150 runtime test passed:
  - Found 1 iluvatar devices
  - test_runtime.py --device iluvatar passed
- All Iluvatar op tests passed:
  - add
  - argmax
  - embedding
  - linear
  - rms_norm
  - rope
  - self_attention
  - swiglu

## Platform Status

- CPU: supported and tested.
- NVIDIA: code path implemented, pending real CUDA machine validation.
- Iluvatar: runtime and op fallback tests passed on Gitee BI-V150.
- Muxi: plan documented, not executed due to unavailable compute.
```

