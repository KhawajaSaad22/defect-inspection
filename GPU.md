# GPU execution

For my RTX 4060, use the optional Linux GPU configuration:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec app python gpu_check.py
```

This builds a distinct local CUDA-capable image instead of pulling the published
CPU image. It requests NVIDIA GPU passthrough and fails startup if CUDA computation
is unavailable. The base Dockerfile remains CPU-compatible for ordinary CI/hosting.
Windows requires the Docker/WSL2 NVIDIA GPU setup and a compatible host driver.

The override installs Linux PyTorch from PyPI rather than the CPU-only wheel index.
That can download several GB of CUDA dependencies. The selected wheel/runtime must
match the host driver; supplying a GPU alone does not change a CPU-only PyTorch build.
Versions remain compatibility ranges, so record/lock the validated deployment stack.

## GPU workload

DINO feature extraction already accepts --device auto|cpu|cuda. Use --device cuda on fit/evaluate/predict to require the GPU. The current PCA/kNN scoring and metadata operations still use CPU; the override does not silently replace those algorithms.

## Validation status — 13 November 2026

The override parses successfully. GPU visibility and real student-model training were
verified on my RTX 4060 in the annotation-assistant project using the existing Linux
CUDA image (PyTorch 2.15.0+cu132). This repository's complete GPU application image
and its real model path have not been independently executed in this update.
The CPU container and its normal tests remain available. No performance or accuracy
improvement is claimed without measuring the actual model/input workload.
