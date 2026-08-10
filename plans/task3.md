# 任务 3 — Qwen2 LLM 推理实现

## 任务本质

这是整个作业的终极目标：**把已经实现的所有 ops 串联起来，构建一个完整的 Qwen2 大语言模型推理引擎**。

前面实现的 `argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu` 都是积木块，任务 3 就是把这些积木块拼成一个能运行真实 LLM 推理的系统。

具体来说，需要：
1. **C++ 后端**：实现 `LlaisysQwen2Model` 结构体，负责模型权重管理、KV-Cache 管理和推理逻辑（Transformer 前向传播）
2. **C API 导出**：实现 `qwen2.h` 定义的 4 个 API：Create、Destroy、Weights、Infer
3. **Python 包装层**：实现 `qwen2.py` 中的 `Qwen2.__init__`（加载权重）和 `Qwen2.generate`（生成文本）

核心难点在于：**KV-Cache**——每次 decode 步骤只计算一个 token 的 Q，但需要访问所有历史 token 的 K 和 V。需要预分配 KV-Cache 张量并在每步推理时追加新的 K/V 行。

## 目标

在 `test/test_infer.py` 中，`llaisys.models.Qwen2` 能通过 argmax 采样生成与 PyTorch 相同的文本（`top_k=1, temperature=1.0`）。

## 背景分析

### 模型参数（DeepSeek-R1-Distill-Qwen-1.5B）

```
nlayer  = 28           # Transformer 层数
hs      = 1536         # hidden size
nh      = 12           # attention heads
nkvh    = 2            # kv heads (GQA)
dh      = 128          # head dim (= hs / nh)
di      = 8960         # intermediate size (FFN)
maxseq  = 131072       # max seq len (可设小值用于测试)
voc     = 151936       # vocab size
epsilon = 1e-6         # rms norm epsilon
theta   = 10000.0      # rope theta
end_token = 151643     # EOS token id
dtype   = bf16
```

### Qwen2 前向传播流程

```
prefill:  token_ids[0..N] → 每次并行处理 N 个 token
decode:   每次处理 1 个 token，使用 KV-Cache

对每个 transformer 层 l:
1. residual = x
2. x = rms_norm(x, attn_norm_w[l], epsilon)
3. q = linear(x, attn_q_w[l], attn_q_b[l])      # [seq, nh, dh]
4. k = linear(x, attn_k_w[l], attn_k_b[l])      # [seq, nkvh, dh]
5. v = linear(x, attn_v_w[l], attn_v_b[l])      # [seq, nkvh, dh]
6. rope(q, q, pos_ids, theta)
7. rope(k, k, pos_ids, theta)
8. 更新 kv_cache: cache_k[l][total:total+seq] = k; cache_v[l][...] = v
9. attn_val = self_attention(q, cache_k[l][:total+seq], cache_v[l][:total+seq], scale)
10. x = linear(attn_val_flat, attn_o_w[l], bias=None)
11. x = x + residual   # 第一个残差
12. residual = x
13. x = rms_norm(x, mlp_norm_w[l], epsilon)
14. gate = linear(x, mlp_gate_w[l], None)
15. up   = linear(x, mlp_up_w[l], None)
16. x = swiglu(gate, up)
17. x = linear(x, mlp_down_w[l], None)
18. x = x + residual   # 第二个残差

最终:
x = rms_norm(x[-1], out_norm_w, epsilon)  # 只取最后一个 token
logits = linear(x, out_embed, None)       # [1, voc]
next_token = argmax(logits)
```

### 权重名称映射（safetensors → 结构体字段）

```python
# 全局
"model.embed_tokens.weight"              → in_embed
"lm_head.weight"                         → out_embed
"model.norm.weight"                      → out_norm_w
# 每层 l
"model.layers.{l}.input_layernorm.weight"            → attn_norm_w[l]
"model.layers.{l}.self_attn.q_proj.weight"           → attn_q_w[l]
"model.layers.{l}.self_attn.q_proj.bias"             → attn_q_b[l]
"model.layers.{l}.self_attn.k_proj.weight"           → attn_k_w[l]
"model.layers.{l}.self_attn.k_proj.bias"             → attn_k_b[l]
"model.layers.{l}.self_attn.v_proj.weight"           → attn_v_w[l]
"model.layers.{l}.self_attn.v_proj.bias"             → attn_v_b[l]
"model.layers.{l}.self_attn.o_proj.weight"           → attn_o_w[l]
"model.layers.{l}.post_attention_layernorm.weight"   → mlp_norm_w[l]
"model.layers.{l}.mlp.gate_proj.weight"              → mlp_gate_w[l]
"model.layers.{l}.mlp.up_proj.weight"                → mlp_up_w[l]
"model.layers.{l}.mlp.down_proj.weight"              → mlp_down_w[l]
```

### 涉及文件

| 文件 | 操作 |
|------|------|
| `src/llaisys/models/qwen2.cc` | **新建** — C++ 模型实现（核心） |
| `src/llaisys/ops.cc` | 不需要修改（已有所有 ops） |
| `python/llaisys/libllaisys/__init__.py` | **修改** — 注册 qwen2 C API |
| `python/llaisys/models/qwen2.py` | **修改** — Python 包装层 |
| `xmake.lua` | **修改** — 把 qwen2.cc 加入编译 |

### 关键 KV-Cache 设计

```
prefill  阶段：seqlen = N（一次处理所有输入 token）
decode   阶段：seqlen = 1（每次处理 1 个 token）

kv_cache_k[l] = tensor([max_cache_len, nkvh, dh], dtype)  # 预分配
kv_cache_v[l] = tensor([max_cache_len, nkvh, dh], dtype)

total_len 追踪当前已填充的 kv 长度
每步 decode 后 total_len += 1（或 += N 在 prefill 后）
```

### rearrange op（已存在）

当 linear 输出是 `[seq, hs]` 而下一步需要 `[seq, nh, dh]` 时，需要用 `view` 重塑，或者直接分配目标形状的输出张量。

---

## 实现计划

### 文件 1：`src/llaisys/models/qwen2.cc`（新建）

C++ 实现核心：

```cpp
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    // KV-Cache：每层一对 [max_cache_len, nkvh, dh] 张量
    std::vector<tensor_t> kv_cache_k;
    std::vector<tensor_t> kv_cache_v;
    size_t total_kv_len = 0;   // 当前 kv-cache 已填充长度

    // 中间缓冲张量（按最大 seqlen 预分配）
    // x, residual, normed, q, k, v, attn_val, gate, up, ...

    llaisysDeviceType_t device;
    int device_id;
};
```

主要函数：
- `llaisysQwen2ModelCreate`：分配 weights 数组（per-layer），分配 kv-cache，初始化中间缓冲区
- `llaisysQwen2ModelInfer(model, token_ids, ntoken)`：
  - embedding(token_ids) → x[seqlen, hs]
  - 生成 pos_ids[seqlen] = [total_kv_len, total_kv_len+1, ..., total_kv_len+seqlen-1]
  - 对每层 l 执行 Transformer forward
  - 最后取 x[-1] 做 rms_norm + lm_head linear + argmax
  - 更新 total_kv_len += seqlen
  - 返回 next_token_id

### 文件 2：`python/llaisys/models/qwen2.py`（修改）

```python
class Qwen2:
    def __init__(self, model_path, device=DeviceType.CPU):
        # 1. 读取 config.json 获取模型超参数
        # 2. 创建 LlaisysQwen2Meta
        # 3. 调用 llaisysQwen2ModelCreate
        # 4. 通过 llaisysQwen2ModelWeights 获取 weights 指针
        # 5. 遍历 safetensors 文件，把每个权重 load 到对应张量

    def generate(self, inputs, max_new_tokens, top_k, top_p, temperature):
        # prefill: 一次性推理所有输入 token
        # decode loop: 每次调用 llaisysQwen2ModelInfer(model, [last_token], 1)
        # 直到 next_token == end_token 或达到 max_new_tokens
        # 返回 inputs + generated_tokens（与 hf 输出格式一致）
```

### 文件 3：`python/llaisys/libllaisys/__init__.py`（修改）

注册 qwen2 相关 C API（与现有 tensor/ops 注册方式相同）。

### 文件 4：`xmake.lua`（修改）

在 `llaisys` target 中加入 `src/llaisys/models/qwen2.cc`。

---

## 实现顺序

1. 修改 `xmake.lua`，让 `src/llaisys/models/*.cc` 加入编译
2. 新建 `src/llaisys/models/qwen2.cc`，实现 C++ 后端
3. 修改 `python/llaisys/libllaisys/__init__.py`，注册 C API
4. 修改 `python/llaisys/models/qwen2.py`，实现 Python 包装层

## 验证方式

```bash
python test/test_infer.py --model <path_to_model> --test --max_steps 10
```

`--test` 模式使用 `top_k=1, temperature=1.0`（greedy），应与 HuggingFace 输出完全一致（token for token）。
