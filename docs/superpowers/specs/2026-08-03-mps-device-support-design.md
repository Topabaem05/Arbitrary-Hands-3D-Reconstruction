# MPS Device Support Design

## Goal

Allow the existing ACR demo entry point to run on Apple Silicon through PyTorch MPS while preserving the original CUDA path and retaining CPU as a final fallback.

## Scope

Only `acr/main.py` changes in production code. It selects `cuda`, then `mps`, then `cpu`; loads checkpoints onto CPU before device placement; moves nested input tensors to the selected device; disables FP16 on non-CUDA devices; and redirects legacy tensor `.cuda()` calls in imported inference modules to the selected non-CUDA device. CUDA keeps `nn.DataParallel` and the existing mixed-precision behavior.

## Error handling

Unsupported MPS operators may use PyTorch's documented CPU fallback through `PYTORCH_ENABLE_MPS_FALLBACK=1`. The selected backend and any precision downgrade are logged. Missing model checkpoints continue to fail closed.

## Validation

Standard-library unit tests validate backend precedence, recursive tensor movement, the non-CUDA compatibility redirect, initialization ordering, CUDA-only DataParallel, and source compilation. The execution environment has no Apple GPU, so full MPS inference remains a hardware validation step rather than a claimed test result.
