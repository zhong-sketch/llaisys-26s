# 结果记录 4.3 - 天数智芯 ILUVATAR 后端接入

## 任务本质

本次执行的目标不是在本机完成真实天数智芯 GPU kernel，而是先把 LLAISYS 的工程结构扩展成可以容纳天数后端的样子：设备枚举、Python 设备名、Xmake 开关、Runtime 分发、算子分发都单独走 `iluvatar` 路径。

真实天数 SDK、编译器、头文件、库路径和 kernel API 需要等进入 Gitee 天数实例后再替换当前占位实现。

## 冲突检查结论

NVIDIA 与天数计划可以同时保留在仓库中，不冲突。关键原因是二者使用完全独立的设备值、宏、开关和源码目录：

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

推荐不要在普通环境中同时打开两个 GPU 后端。分别使用：

```bash
# NVIDIA 环境
xmake f --nv-gpu=y --iluvatar-gpu=n -cv

# 天数智芯环境
xmake f --nv-gpu=n --iluvatar-gpu=y -cv
```

## 修改/新建文件

| 文件 | 操作 | 作用 |
|---|---|---|
| `include/llaisys.h` | 修改 | 新增 `LLAISYS_DEVICE_ILUVATAR = 2` |
| `python/llaisys/libllaisys/llaisys_types.py` | 修改 | 新增 Python 侧 `DeviceType.ILUVATAR` |
| `python/llaisys/models/qwen2.py` | 修改 | 支持字符串设备名 `"iluvatar"` |
| `test/test_utils.py` | 修改 | 支持测试脚本将 `"iluvatar"` 映射到 LLAISYS 设备类型 |
| `test/test_runtime.py`、`test/test_infer.py`、`test/ops/*.py` | 修改 | argparse 设备参数增加 `"iluvatar"` |
| `src/device/runtime_api.hpp` | 修改 | 声明 `iluvatar::getRuntimeAPI()` |
| `src/device/runtime_api.cpp` | 修改 | 新增 `LLAISYS_DEVICE_ILUVATAR` Runtime 分发 |
| `xmake.lua` | 修改 | 新增 `--iluvatar-gpu` 编译开关 |
| `xmake/iluvatar.lua` | 新建 | 定义天数 device/ops target |
| `src/device/iluvatar/` | 新建 | 天数 Runtime 占位实现 |
| `src/ops/iluvatar/unsupported.hpp` | 新建 | 天数算子占位错误提示 |
| `src/ops/*/iluvatar/` | 新建 | 8 个算子的天数后端入口 |
| `src/ops/*/op.cpp` | 修改 | 新增 `LLAISYS_DEVICE_ILUVATAR` 分发分支 |
| `Result/res4.md` | 修改 | 补充 NVIDIA 计划与天数计划的隔离关系 |

## 代码变化

### 1. 新增设备类型

修改后：

```cpp
typedef enum {
    LLAISYS_DEVICE_CPU = 0,
    LLAISYS_DEVICE_NVIDIA = 1,
    LLAISYS_DEVICE_ILUVATAR = 2,
    LLAISYS_DEVICE_TYPE_COUNT
} llaisysDeviceType_t;
```

含义：`ILUVATAR` 不复用 `NVIDIA` 的值，而是作为第三类设备进入统一 Runtime API。

### 2. 新增 Xmake 开关

修改后：

```lua
option("iluvatar-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Iluvatar GPU")
option_end()

if has_config("iluvatar-gpu") then
    add_defines("ENABLE_ILUVATAR_API")
    includes("xmake/iluvatar.lua")
end
```

含义：默认不编译天数后端；只有显式传入 `--iluvatar-gpu=y` 时，才会打开 `ENABLE_ILUVATAR_API` 并加入天数 target。

### 3. 新增 Runtime 分发

修改后：

```cpp
case LLAISYS_DEVICE_ILUVATAR:
#ifdef ENABLE_ILUVATAR_API
    return llaisys::device::iluvatar::getRuntimeAPI();
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
```

含义：上层看到 `DeviceType.ILUVATAR` 后，会进入天数 Runtime API；如果编译时没有启用天数后端，则保持“不支持该设备”的明确错误。

### 4. 新增算子分发

以 `add` 为例：

```cpp
#ifdef ENABLE_ILUVATAR_API
case LLAISYS_DEVICE_ILUVATAR:
    return iluvatar::add(c->data(), a->data(), b->data(),
                         c->dtype(), c->numel());
#endif
```

含义：调度层只负责把 `add` 分发到对应设备。当前 `iluvatar::add` 是占位实现，后续在天数实例中替换为真实天数 kernel。

## 当前实现状态

已完成：

1. CPU、NVIDIA、ILUVATAR 三套设备类型不互相覆盖。
2. `--nv-gpu` 和 `--iluvatar-gpu` 是独立 Xmake 开关。
3. Python 测试脚本可以接受 `--device iluvatar`。
4. 本地 `--iluvatar-gpu=y` 可以编译通过，占位后端会被正确纳入工程。
5. 本地无天数设备时，`test_runtime.py --device iluvatar` 能正常查询并跳过。

暂未完成：

1. 未接入真实天数 SDK。
2. 未实现真实天数显存分配、拷贝、stream、device context。
3. 8 个天数算子目前都是占位实现，会提示需要先配置天数 SDK。
4. 还不能证明 `--device iluvatar` 的真实计算结果与 PyTorch 对齐。

## 验证结果

### 1. 空白检查

```powershell
git diff --check
```

结果：无空白错误；仅有 Windows 换行提示。

### 2. CPU 默认路径验证

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=n --iluvatar-gpu=n -cv
D:\LLAISYS\env\xmake\xmake.exe
D:\LLAISYS\env\xmake\xmake.exe install
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'
D:\LLAISYS\env\python\.venv\Scripts\python.exe test/test_runtime.py --device cpu
```

结果：

```text
build ok
install ok
Found 1 cpu devices
Test passed!
```

### 3. 天数占位后端编译验证

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=n --iluvatar-gpu=y -cv
D:\LLAISYS\env\xmake\xmake.exe
D:\LLAISYS\env\xmake\xmake.exe install
$env:PYTHONPATH='D:\LLAISYS\code\llaisys-26s\python'
D:\LLAISYS\env\python\.venv\Scripts\python.exe test/test_runtime.py --device iluvatar
```

结果：

```text
build ok
install ok
Found 0 iluvatar devices
Skipped
Test passed!
```

解释：这说明 `iluvatar` 设备入口、Python 参数、Runtime 分发都已经连上；但因为本机没有天数硬件和 SDK，所以设备数为 0，测试只验证了入口不崩溃，不代表真实天数计算已完成。

### 4. 恢复默认 CPU 配置

最后已执行：

```powershell
D:\LLAISYS\env\xmake\xmake.exe f --nv-gpu=n --iluvatar-gpu=n -cv
D:\LLAISYS\env\xmake\xmake.exe
D:\LLAISYS\env\xmake\xmake.exe install
```

结果：

```text
build ok
install ok
```

## 后续在 Gitee 天数实例中要做的事

1. 进入天数算力实例后确认真实工具链命令，例如 `ixsmi`、`ixcc` 或平台文档指定名称。
2. 将 `src/device/iluvatar/iluvatar_runtime_api.cpp` 从占位实现替换成真实 Runtime API。
3. 将 `src/ops/*/iluvatar/*.cpp` 从占位实现替换成真实天数 kernel 或兼容 API 调用。
4. 运行全部 `test/ops/*.py --device iluvatar`。
5. 最后运行 `test/test_infer.py --device iluvatar --test --max_steps 1`，和 PyTorch 结果对齐。

## 结论

本次已经完成天数智芯后端的工程接入骨架，并确认它不会和 NVIDIA 后端冲突。当前成果属于“可编译、可分发、可进入测试入口”的占位版本；真正的天数算力验证需要等 Gitee 天数实例可用后继续完成。
