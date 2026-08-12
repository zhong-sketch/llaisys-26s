## Summary

- Complete the NVIDIA CUDA backend path for LLAISYS Runtime API, xmake build integration, GPU operator dispatch, and Qwen2 GPU inference.
- Add Iluvatar device integration with an independent device type, runtime dispatch, xmake option, Python/test device mapping, and correctness-first operator fallback path.
- Keep NVIDIA and Iluvatar builds isolated with separate switches: `--nv-gpu` and `--iluvatar-gpu`.
- Fix Windows CUDA DLL discovery, runtime lazy initialization, runtime destruction, NVIDIA fp16 conversion, and the self-attention CUDA test mask device.

## Verification

- CPU:
  - `test/test_runtime.py --device cpu`
  - `test/test_tensor.py`
  - all 8 op tests with `--device cpu`
  - `test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device cpu`
- NVIDIA, local RTX 4070 Laptop GPU + CUDA 12.8:
  - `test/test_runtime.py --device nvidia`
  - all 8 op tests with `--device nvidia`
  - `test/test_infer.py --model D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 1 --device nvidia`
- Iluvatar, Gitee BI-V150 instance:
  - `test/test_runtime.py --device iluvatar`
  - all 8 op tests with `--device iluvatar`

## Platform Status

| Platform | Status |
|---|---|
| CPU | Runtime, tensor tasks, CPU operators, and Qwen2 inference passed. |
| NVIDIA | Runtime, CUDA build, 8 GPU operators, and Qwen2 GPU inference passed on RTX 4070 Laptop GPU. |
| Iluvatar | Runtime and 8 operator tests passed on Gitee BI-V150; operators use correctness-first D2H/CPU/H2D fallback. |
| Muxi | Plan documented only; not executed because compute was unavailable. |

## Notes

- GitHub Actions currently validates CPU regression only because the hosted runners do not provide NVIDIA or Iluvatar GPUs.
- NVIDIA GPU verification was performed locally.
- Iluvatar verification was performed on a rented Gitee compute instance.
