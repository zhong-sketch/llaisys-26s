# 任务 4.1-NVIDIA - NVIDIA CUDA 算力接入计划

## 任务本质

NVIDIA 是标准 CUDA 平台。本计划只负责 `nvidia` 后端，不承担天数、沐曦等国产平台的实现，避免不同 SDK 和编译环境互相覆盖。

NVIDIA 后端在系统中的位置：

```text
DeviceType.NVIDIA
  -> LLAISYS_DEVICE_NVIDIA
  -> ENABLE_NVIDIA_API
  -> xmake/nvidia.lua
  -> src/device/nvidia/
  -> src/ops/*/nvidia/
```

## 与天数计划的隔离规则

1. NVIDIA 固定使用枚举值 `1`：

```cpp
LLAISYS_DEVICE_NVIDIA = 1
```

2. NVIDIA 只使用宏：

```cpp
ENABLE_NVIDIA_API
```

3. NVIDIA 只使用 Xmake 开关：

```bash
xmake f --nv-gpu=y
```

4. NVIDIA 代码只放在：

```text
src/device/nvidia/
src/ops/*/nvidia/
```

5. 不在 NVIDIA 文件中引用 `iluvatar` 命名空间、天数 SDK 头文件或天数库。

6. 不建议在同一台机器上同时打开：

```bash
xmake f --nv-gpu=y --iluvatar-gpu=y
```

除非该机器同时安装 CUDA SDK 和天数 SDK，并确认两套运行时库不会冲突。

推荐编译方式：

```bash
# NVIDIA 环境
xmake f --nv-gpu=y --iluvatar-gpu=n -cv
```

## 目标

1. 使用 `LLAISYS_DEVICE_NVIDIA` 作为设备类型。
2. 使用 `ENABLE_NVIDIA_API` 控制编译。
3. 使用 `xmake/nvidia.lua` 编译 NVIDIA Runtime 和 NVIDIA 算子。
4. 通过 `--device nvidia` 测试 Runtime、算子和 Qwen2 推理。

## 验证命令

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

xmake f --nv-gpu=y --iluvatar-gpu=n -cv
xmake
xmake install

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
```

完整推理：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --test \
  --max_steps 1
```

## 当前状态

NVIDIA 代码路径已经在作业 #4 中实现，但当前环境没有 CUDA SDK，Gitee NVIDIA 算力也暂时售罄，因此暂时无法完成 NVIDIA 运行验证。
