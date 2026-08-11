## Summary

- Add NVIDIA CUDA backend scaffolding and operator implementations.
- Add Iluvatar device type, runtime dispatch, xmake option, and Python test support.
- Implement Iluvatar CUDA-compatible runtime using CoreX CUDA runtime paths.
- Add correctness-first Iluvatar operator fallback implementations for add, argmax, embedding, linear, rms_norm, rope, self_attention, and swiglu.

## Verification

- CPU runtime test passed locally.
- Gitee Iluvatar BI-V150 runtime test passed:
  - Found 1 iluvatar devices
  - `test_runtime.py --device iluvatar` passed
- All Iluvatar op tests passed:
  - `add`
  - `argmax`
  - `embedding`
  - `linear`
  - `rms_norm`
  - `rope`
  - `self_attention`
  - `swiglu`

## Platform Status

- CPU: supported and tested.
- NVIDIA: code path implemented, pending real CUDA machine validation.
- Iluvatar: runtime and op fallback tests passed on Gitee BI-V150.
- Muxi: plan documented, not executed due to unavailable compute.

