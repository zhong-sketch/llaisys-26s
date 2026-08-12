---
name: llaisys-26s_task
description: 处理 llaisys-26s 项目的编程任务。用户上传任务截图或作业要求后，先生成实现计划（含任务本质解释），等待确认后执行，并记录代码变更、验证结果和提交报告。
---

# llaisys-26s 任务处理工作流

本技能用于处理 `llaisys-26s` 项目的作业、代码实现、调试验证和提交报告整理。

## 阶段一：生成实现计划（先不执行）

当用户上传任务截图、作业说明或要求“先不执行”时，必须先生成计划，不能直接改代码。

1. **阅读并理解任务**
   - 提取任务编号，例如 `1.1`、`2.4`、`3`、`4.1`。
   - 提取函数签名、目标文件、设备平台、测试命令和提交要求。
   - 用自己的话解释任务本质，不照抄截图。

2. **研究代码库**
   - 阅读相关源文件，理解现有代码结构和命名模式。
   - 优先查找 `TO_BE_IMPLEMENTED()`、占位实现、已有 CPU/NVIDIA/ILUVATAR 类似实现。
   - 识别 C API、Python ctypes、C++ 调度层、Runtime API、算子后端之间的调用关系。

3. **生成计划文件**
   - 计划文件放在：

```text
D:\LLAISYS\code\llaisys-26s\plans\task<编号>.md
```

   - 内容结构：

```markdown
# 任务 <编号> - `<函数或目标>`

## 任务本质

说明这个任务本质上要做什么、为什么需要它、它在 LLAISYS 架构中扮演什么角色。

## 目标

列出本任务完成后应该具备的能力。

## 背景分析

### 涉及文件

列出相关文件及其作用。

### 关键 API

列出相关 C API、C++ 函数、Python 包装、Runtime API 或测试入口。

### 参考模式

列出仓库里可参考的已有实现。

## 实现计划

### 修改位置

列出计划修改的文件和大致位置。

### 实现逻辑

说明准备如何实现，不需要在计划阶段直接改代码。

### 风险点

说明平台、dtype、内存布局、设备上下文、CI 或测试环境风险。

## 验证方式

列出本地测试、GPU 实机测试、GitHub Actions 或 PR 检查方式。
```

4. **等待用户确认**
   - 计划生成后明确说明：计划已生成，等待用户确认后再执行。
   - 用户明确要求执行后，才进入阶段二。

## 阶段二：执行计划并记录变更

用户确认执行后：

1. **执行代码修改**
   - 按计划修改源文件。
   - 使用项目已有模式，避免引入不必要的新抽象。
   - 不还原用户已有改动，不混入无关重构。

2. **运行验证**
   - 优先运行与任务直接相关的测试。
   - 对影响范围大的改动，补充 CPU 回归、GPU 设备测试或推理测试。
   - 记录具体命令和结果。

3. **生成结果记录**
   - 结果文件放在：

```text
D:\LLAISYS\code\llaisys-26s\Result\res<编号>.md
```

   - 内容结构：

```markdown
# 结果记录 <编号> - `<函数或目标>`

## 修改文件

列出修改或新增的文件。

## 代码变化

### 修改前

展示关键旧代码或占位实现。

### 修改后

展示关键新代码。

### Diff

用 diff 形式说明核心变化。

## 逻辑解释

解释关键代码为什么这样写。

## 验证结果

记录实际执行的测试命令和结果。
```

## 作业 #4 / 平台接入类任务补充规则

当任务涉及 NVIDIA、天数智芯 ILUVATAR、沐曦 MUXI、摩尔线程或其他 GPU/类 CUDA 平台时：

1. **平台必须隔离**
   - 不同平台使用独立设备枚举、编译宏、Xmake 开关和源码目录。
   - 示例：

```text
NVIDIA:
  LLAISYS_DEVICE_NVIDIA
  ENABLE_NVIDIA_API
  --nv-gpu
  src/device/nvidia/
  src/ops/*/nvidia/

ILUVATAR:
  LLAISYS_DEVICE_ILUVATAR
  ENABLE_ILUVATAR_API
  --iluvatar-gpu
  src/device/iluvatar/
  src/ops/*/iluvatar/
```

2. **报告必须逐平台说明**
   - CPU 状态。
   - NVIDIA 状态。
   - 天数/沐曦/摩尔等国产平台状态。
   - 未执行的平台必须诚实说明原因。

3. **不要夸大验证结果**
   - 原生 GPU kernel、CUDA 兼容 Runtime、D2H/CPU/H2D fallback 是不同完成度。
   - 如果算子是 fallback，必须在报告里明确写明。

4. **提交报告位置**
   - 作业 #4 报告：

```text
submission_reports/task4/README.md
submission_reports/task4/pr_description.md
```

   - 最终总报告：

```text
submission_reports/final/README.md
submission_reports/final/pr_description.md
```

5. **PR 提交要求**
   - CI 必须通过。
   - PR 描述或 Markdown 报告必须包含复现流程和复现结果。
   - 必须逐个平台说明支持状态。
   - 如果 GitHub Actions 只覆盖 CPU，需要说明 GPU 由本地或云平台实机验证。

## 重要规则

- 任务编号从用户截图或上下文中提取。
- 先计划、后执行；用户明确要求直接执行时才跳过等待。
- 计划必须包含“任务本质”。
- 结果必须包含修改内容、逻辑解释和验证结果。
- 所有路径使用本项目路径：

```text
D:\LLAISYS\code\llaisys-26s
```
