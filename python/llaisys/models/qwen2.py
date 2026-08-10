import ctypes
import json
from pathlib import Path
from typing import Sequence

import safetensors
import torch

from ..libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
)
from ..tensor import Tensor


def _dtype_from_torch(dtype):
    if dtype == torch.float32:
        return DataType.F32
    if dtype == torch.float16:
        return DataType.F16
    if dtype == torch.bfloat16:
        return DataType.BF16
    if dtype == torch.int64:
        return DataType.I64
    raise TypeError(f"Unsupported model weight dtype: {dtype}")


def _dtype_from_config(config):
    name = config.get("torch_dtype", "bfloat16")
    if name in ("float32", "float"):
        return DataType.F32
    if name in ("float16", "half"):
        return DataType.F16
    if name in ("bfloat16", "bf16"):
        return DataType.BF16
    raise TypeError(f"Unsupported model torch_dtype: {name}")


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        if isinstance(device, str):
            device = DeviceType.NVIDIA if device == "nvidia" else DeviceType.CPU

        self.model_path = Path(model_path)
        self.device = DeviceType(device)
        self.device_id = 0
        self._model = None
        self._weights_ptr = None
        self._weight_tensors = []

        with (self.model_path / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim", hidden_size // num_heads))

        # Keep the test-time cache bounded. It can grow on demand up to this limit.
        maxseq = min(int(config["max_position_embeddings"]), 4096)
        meta = LlaisysQwen2Meta(
            dtype=_dtype_from_config(config),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=head_dim,
            di=int(config["intermediate_size"]),
            maxseq=maxseq,
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config["rope_theta"]),
            end_token=int(config["eos_token_id"]),
        )
        self._meta = meta
        self._end_token = int(config["eos_token_id"])

        device_ids = (ctypes.c_int * 1)(self.device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            int(self.device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create LLAISYS Qwen2 model")

        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._load_weights(config)

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def _make_weight_tensor(self, tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        llaisys_tensor = Tensor(
            tuple(int(dim) for dim in tensor.shape),
            dtype=_dtype_from_torch(tensor.dtype),
            device=self.device,
            device_id=self.device_id,
        )
        llaisys_tensor.load(ctypes.c_void_p(int(tensor.data_ptr())))
        self._weight_tensors.append(llaisys_tensor)
        return llaisys_tensor.lib_tensor()

    def _set_weight(self, name, tensor):
        weights = self._weights_ptr.contents

        if name == "model.embed_tokens.weight":
            weights.in_embed = self._make_weight_tensor(tensor)
            return
        if name == "lm_head.weight":
            weights.out_embed = self._make_weight_tensor(tensor)
            return
        if name == "model.norm.weight":
            weights.out_norm_w = self._make_weight_tensor(tensor)
            return

        prefix = "model.layers."
        if not name.startswith(prefix):
            raise KeyError(f"Unexpected Qwen2 weight: {name}")

        remainder = name[len(prefix):]
        layer_text, field_name = remainder.split(".", 1)
        layer = int(layer_text)
        mapping = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }
        if field_name not in mapping:
            raise KeyError(f"Unexpected Qwen2 layer weight: {name}")

        field = mapping[field_name]
        getattr(weights, field)[layer] = self._make_weight_tensor(tensor)

    def _load_weights(self, config):
        expected_dtype = _dtype_from_config(config)
        loaded = 0

        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"No safetensors files found in {self.model_path}"
            )

        # framework="pt" is used only to read bf16 storage safely. Inference
        # itself remains entirely inside the LLAISYS C++ backend.
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as data:
                for name in data.keys():
                    tensor = data.get_tensor(name)
                    if _dtype_from_torch(tensor.dtype) != expected_dtype:
                        raise TypeError(
                            f"Unexpected dtype for {name}: {tensor.dtype}"
                        )
                    self._set_weight(name, tensor)
                    loaded += 1

        expected = 3 + self._meta.nlayer * 12
        if loaded != expected:
            raise RuntimeError(
                f"Loaded {loaded} Qwen2 tensors, expected {expected}"
            )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        del top_k, top_p, temperature

        tokens = [int(token) for token in inputs]
        if not tokens:
            raise ValueError("Qwen2.generate requires at least one input token")

        steps = 128 if max_new_tokens is None else int(max_new_tokens)
        if steps <= 0:
            return tokens

        feed = tokens
        for _ in range(steps):
            token_array = (ctypes.c_int64 * len(feed))(*feed)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_array, len(feed)
                )
            )
            tokens.append(next_token)
            if next_token == self._end_token:
                break
            feed = [next_token]

        return tokens
