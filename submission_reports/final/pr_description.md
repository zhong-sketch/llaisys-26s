## Summary

- Complete LLAISYS homework #1 tensor APIs: load, contiguous check, view, permute, and slice.
- Complete homework #2 CPU operators: add, argmax, embedding, linear, rms_norm, rope, self_attention, and swiglu.
- Complete homework #3 Qwen2 inference path with Python ctypes bindings, C API, C++ model forward, safetensors loading, and KV cache.
- Complete homework #4 NVIDIA CUDA backend and Iluvatar backend integration.

## Homework #4 Platform Coverage

| Platform | Status |
|---|---|
| CPU | Runtime, tensor tests, CPU operators, and Qwen2 inference passed. |
| NVIDIA | Runtime, CUDA build, 8 GPU operators, and Qwen2 GPU inference passed on RTX 4070 Laptop GPU. |
| Iluvatar | Runtime and 8 operator tests passed on Gitee BI-V150; operators use correctness-first D2H/CPU/H2D fallback. |
| Muxi | Plan documented only; not executed because compute was unavailable. |

## Verification

CPU:

- `python test/test_runtime.py --device cpu`
- `python test/test_tensor.py`
- all `test/ops/*.py --device cpu`
- `python test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device cpu`

NVIDIA:

- `xmake f --nv-gpu=y --iluvatar-gpu=n -cv`
- `xmake`
- `xmake install`
- `python test/test_runtime.py --device nvidia`
- all `test/ops/*.py --device nvidia`
- `python test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device nvidia`

Iluvatar:

- `xmake f --nv-gpu=n --iluvatar-gpu=y -cv`
- `xmake`
- `xmake install`
- `python test/test_runtime.py --device iluvatar`
- all `test/ops/*.py --device iluvatar`

## Notes

- CPU and NVIDIA Qwen2 inference both matched HuggingFace/PyTorch token output:
  `[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]`
- GitHub Actions currently validates CPU regression only because hosted runners do not provide NVIDIA or Iluvatar GPUs.
- Detailed reproduction notes are included in `submission_reports/final/README.md` and `submission_reports/task4/README.md`.
- The pull request is intended for `wooway777/llaisys-26s`, from `zhong-sketch:main`.
