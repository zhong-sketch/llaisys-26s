# 任务 4.3-ILUVATAR - 天数智芯国产算力接入计划

## 任务本质

天数智芯是当前 Gitee 算力市场仍有库存的国产算力平台。本计划目标是在不破坏 CPU 和 NVIDIA 后端的前提下，为 LLAISYS 新增并列的 `iluvatar` 后端。

天数后端在系统中的位置：

```text
DeviceType.ILUVATAR
  -> LLAISYS_DEVICE_ILUVATAR
  -> ENABLE_ILUVATAR_API
  -> xmake/iluvatar.lua
  -> src/device/iluvatar/
  -> src/ops/*/iluvatar/
```

## 与 NVIDIA 计划的隔离规则

1. 天数使用独立枚举值：

```cpp
LLAISYS_DEVICE_ILUVATAR = 2
```

2. 天数只使用宏：

```cpp
ENABLE_ILUVATAR_API
```

3. 天数只使用 Xmake 开关：

```bash
xmake f --iluvatar-gpu=y
```

4. 天数代码只放在：

```text
src/device/iluvatar/
src/ops/*/iluvatar/
```

5. 天数调度分支使用独立命名空间：

```cpp
llaisys::ops::iluvatar
llaisys::device::iluvatar
```

6. 推荐在天数实例中只打开天数后端：

```bash
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
```

## 目标

1. 新增 `LLAISYS_DEVICE_ILUVATAR`。
2. 新增 Python `DeviceType.ILUVATAR`。
3. 新增测试参数 `"iluvatar"`。
4. 新增 `ENABLE_ILUVATAR_API` 和 `--iluvatar-gpu`。
5. 新增天数 Runtime 和算子隔离目录。
6. 本地先保证 CPU 构建不回归；天数 SDK 相关编译在 Gitee 天数实例里验证。

## 涉及文件

| 文件 | 计划操作 |
|---|---|
| `include/llaisys.h` | 新增 `LLAISYS_DEVICE_ILUVATAR = 2` |
| `python/llaisys/libllaisys/llaisys_types.py` | 新增 `DeviceType.ILUVATAR = 2` |
| `python/llaisys/models/qwen2.py` | 识别字符串 `"iluvatar"` |
| `test/test_utils.py` | 新增 `"iluvatar"` 映射 |
| `test/test_runtime.py` | argparse 允许 `"iluvatar"` |
| `test/test_infer.py` | argparse 允许 `"iluvatar"` |
| `test/ops/*.py` | argparse 允许 `"iluvatar"` |
| `src/device/runtime_api.hpp` | 声明 `iluvatar::getRuntimeAPI()` |
| `src/device/runtime_api.cpp` | 分发 `LLAISYS_DEVICE_ILUVATAR` |
| `src/device/iluvatar/` | 新增天数 Runtime |
| `xmake.lua` | 新增 `iluvatar-gpu` option |
| `xmake/iluvatar.lua` | 新增天数编译 target |
| `src/ops/*/iluvatar/` | 新增天数算子入口 |
| `src/ops/*/op.cpp` | 新增 ILUVATAR 分支 |

## 实施步骤

### 第一步：确认 Gitee 天数实例

进入实例后先检查：

```bash
uname -a
python --version
which xmake
which ixsmi
which ixcc
ixsmi
```

如果命令名不同，以 Gitee 实例镜像说明为准。

### 第二步：新增设备类型和编译开关

新增：

```cpp
LLAISYS_DEVICE_ILUVATAR = 2
```

新增：

```lua
option("iluvatar-gpu")
```

### 第三步：新增 Runtime

新增：

```text
src/device/iluvatar/iluvatar_runtime_api.cu
src/device/iluvatar/iluvatar_resource.cu
src/device/iluvatar/iluvatar_resource.cuh
```

在没有天数 SDK 的本地环境中，这些文件只在 `--iluvatar-gpu=y` 时参与编译，因此不会影响 CPU/NVIDIA 路径。

### 第四步：新增算子入口

新增 8 个算子的 `iluvatar` 目录，并先按类 CUDA 接口组织：

```text
src/ops/add/iluvatar/
src/ops/argmax/iluvatar/
src/ops/embedding/iluvatar/
src/ops/linear/iluvatar/
src/ops/rms_norm/iluvatar/
src/ops/rope/iluvatar/
src/ops/self_attention/iluvatar/
src/ops/swiglu/iluvatar/
```

如果天数 SDK 对 CUDA Runtime 兼容，则可以复用 NVIDIA 朴素 kernel 结构；否则需要进入天数实例后按 SDK 改写。

### 第五步：验证

本地先验证：

```bash
xmake f --nv-gpu=n --iluvatar-gpu=n -cv
xmake
python test/test_runtime.py --device cpu
```

天数实例中验证：

```bash
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
xmake
xmake install
python test/test_runtime.py --device iluvatar
```

算子：

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

推理：

```bash
python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device iluvatar \
  --test \
  --max_steps 1
```

## 风险点

1. 本地没有天数 SDK，只能验证隔离设计和 CPU 不回归。
2. 天数真实 SDK 函数名、头文件和链接库需要在 Gitee 实例中确认。
3. 如果 PyTorch 不能直接识别天数设备，后续测试需要改为 CPU PyTorch 生成答案、天数后端执行、D2H 拷回后比较。
