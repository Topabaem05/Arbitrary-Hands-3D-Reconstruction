# MPS Device Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ACR demo select CUDA, Apple MPS, or CPU at runtime without changing the model architecture.

**Architecture:** Keep all device policy in `acr/main.py`. A small compatibility redirect maps legacy tensor `.cuda()` calls inside imported inference modules to the selected non-CUDA backend, while CUDA retains its native implementation and `nn.DataParallel` path.

**Tech Stack:** Python, PyTorch, unittest, AST-based source checks.

## Global Constraints

- Preserve CUDA behavior and CUDA FP16.
- Use FP32 on MPS and CPU.
- Load checkpoints with `map_location='cpu'` before moving modules.
- Do not claim MPS hardware validation from the Linux CPU test environment.

---

### Task 1: Add failing device-policy tests

**Files:**
- Create: `tests/test_mps_main.py`
- Modify: none

- [x] Test CUDA/MPS/CPU selection precedence.
- [x] Test recursive tensor movement.
- [x] Test the legacy `.cuda()` redirect.
- [x] Test source ordering and CUDA-only DataParallel.
- [x] Run `python -m unittest tests/test_mps_main.py -v` and verify failure against the original entry point.

### Task 2: Implement the entry-point device policy

**Files:**
- Modify: `acr/main.py`
- Test: `tests/test_mps_main.py`

- [x] Set MPS fallback before importing PyTorch.
- [x] Add device selection, recursive movement, checkpoint loading, and legacy redirect helpers.
- [x] Force FP32 on MPS/CPU and keep CUDA DataParallel/FP16.
- [x] Move input metadata and modules to the selected device.
- [x] Run unit tests and Python compilation checks.

### Task 3: Publish and verify

**Files:**
- Commit the design, plan, production change, and regression tests.

- [ ] Push branch `agent/mps-device-support`.
- [ ] Open a draft pull request targeting `main`.
- [ ] Fetch the committed file and compare branch against `main`.
