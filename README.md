# Few-shot defect inspection under domain shift

Dataset audit update: `python dataset_audit.py manifest.json --output audit.json`
checks IDs, file readability, masks, class counts, exact duplicates and group overlap.
Each row needs `id`, `image`, `split`, `group`, and either a `mask` or `label: 0`
for a normal image; pool samples may omit labels. Paths are relative to the manifest.
Near-duplicate cross-split candidates use dHash and require human review. For more
than 2,000 samples use `--near-distance -1` to disable the quadratic candidate scan.
Exit code 2 means blocking issues; input files are never changed. Define groups by
related physical objects/capture sessions and run this audit before experiments.

Research question: with only 1–4 normal reference images, how much does defect detection
degrade under illumination changes, and does normal-only augmentation reduce the damage?

Implemented: pretrained DINOv2-small patch features, normal-only PCA residual and kNN
baselines, disjoint held-out normal calibration, fixed-threshold evaluation, pixel maps,
review UI, synthetic smoke dataset, MVTec AD adapter, paired experiment runner.
This is an independent compact baseline inspired by SubspaceAD, not a reproduction of
the paper's numbers or exact pipeline. No upstream implementation was copied.

## Setup

Use a separate environment for this project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CUDA, install the compatible PyTorch build using https://pytorch.org/get-started/locally/
before installing requirements. My existing sibling `../hybrid-rag/.venv` can also run
the scripts; it currently contains CPU-only PyTorch. My RTX 4060 has 8 GB VRAM.
Default DINO input is 224 square pixels; increase only in multiples of 14 after profiling.
First DINO use downloads official Hugging Face model weights. No paid API is needed.

## Run the complete demo

Replace `python` below with my environment's Python executable:

```powershell
python make_demo.py
python inspect_defects.py fit --train data/demo/train --calibration data/demo/calibration --output artifacts/dino-pca.npz
python inspect_defects.py evaluate --model artifacts/dino-pca.npz --manifest data/demo/test.json --output reports/dino-pca.json
python inspect_defects.py predict --model artifacts/dino-pca.npz --image data/demo/test/020.png --output artifacts/overlay.png
python -m streamlit run app.py
python -m unittest -v
```

For a no-download smoke baseline, pass `--backend rgb` during fitting. RGB uses raw
patch pixels, explicitly not learned semantic features. Demo images are deliberately
simple and synthetic; their metrics cannot establish industrial robustness.

## Real data

Download a MVTec AD category yourself under its dataset terms. Then:

```powershell
python prepare_mvtec.py C:/datasets/mvtec/bottle --output data/bottle
python run_experiments.py --data data/bottle --shots 1 4 --seeds 17 42 73 --output experiments/bottle-v1
```

The adapter copies data into new folders, reserves 20 normal training images for
calibration, and leaves source data unchanged. For custom images use separate normal
`train/` and `calibration/` folders plus a test JSON list:

```json
[{"path":"test/good.png","label":0}, {"path":"test/scratch.png","label":1,"mask":"masks/scratch.png"}]
```

Paths are relative to the manifest. Test content hashes are checked against all available
training and calibration images. Split by object and acquisition session yourself: hashes
cannot detect near-duplicates. Optional masks must align with their input image.

## Protocol and interpretation

Fit only on the sampled normal reference images. Estimate a 95th-percentile threshold
on held-out normal calibration images. The 5% target false-alarm rate is a calibration
target, not a statistical guarantee, especially with small calibration sets. Evaluate
with that same threshold under clean, darker, brighter, contrast-reduced, and blurred
images. Photometric shifts keep mask geometry unchanged.

Reports contain image AUROC/AP, fixed-threshold recall/FPR, per-image scores, latency,
and pixel AUROC for the masked subset. Pixel metrics on a subset are not comparable to
full-benchmark pixel metrics. These scripts do not calculate AU-PRO. No test-set fitting,
threshold selection, or adaptive normalization is performed.

The experiment runner compares kNN, PCA, and PCA with fixed photometric augmentation
using the same sampled reference images per seed. PCA retains 95% normal variance;
the feature-bank cap is 4096 patches. DINO features are normalized; RGB features are not.
Input resizing distorts aspect ratio. Top-1% patch pooling is fixed. Extremely subtle
defects below patch resolution may be missed. Heatmap color is relative to normal
calibration pixel-score percentiles (95th to 99.9th), not a calibrated defect probability.

## Next milestones

1. Run the paired experiment on several real product categories; report mean/std across seeds.
2. Add real camera/lighting sessions and hold out entire sessions.
3. Cache features to accelerate ablations, then profile GPU memory and throughput.
4. Study foreground masking and higher resolutions as separately controlled ablations.
5. Distill a compact student and report accuracy versus latency.

Research: https://github.com/CLendering/SubspaceAD
Backbone: https://github.com/facebookresearch/dinov2
Dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
MVTec data is not bundled; its non-commercial dataset terms apply separately.

## Docker image

This project has its own CPU image: `ghcr.io/khawajasaad22/defect-inspection:latest`.
Images are stored in GitHub Container Registry; the repository contains the Dockerfile,
Compose configuration, and build/publish workflow. The workflow tests the image and
checks the running app before publishing `latest` and a full commit-SHA tag. A failed
build does not publish. Use the Actions tab to confirm the first successful publication.

With Docker Engine/Desktop and Compose installed, from this repository:

```sh
docker compose up --build -d
```

Open http://localhost:8502. To use the registry image after it is published:

```sh
docker compose pull app
docker compose up -d --no-build app
```

If the registry package is private, authenticate with `docker login ghcr.io` using an
account/token allowed to read the package. Repository visibility alone does not ensure
anonymous image access. No custom registry secret is needed by the publishing workflow:
it uses the repository's built-in GitHub token with package-write permission.

The container runs as UID 10001. Named volumes preserve inputs, generated artifacts,
and downloaded model weights. Weights and private datasets are not baked into the image.
First model use requires internet access. These are Linux/amd64 CPU images; GPU/CUDA
acceleration is not included. Initial builds can take several minutes.

Use `docker compose exec app python ...` for the documented CLI commands. Inputs are
container paths, not Windows host paths. Copy real files with `docker compose cp`:

```sh
docker compose cp ./my-input app:/app/data/my-input
docker compose logs -f app
docker compose down
```

`down` keeps named volumes; `down -v` deletes them, including my generated work.
The app port is published on localhost only. Container health means Streamlit is running;
I still need to prepare data and fit/index models before using the app workflow.

```sh
docker compose exec app python make_demo.py
docker compose exec app python inspect_defects.py fit --train data/demo/train --calibration data/demo/calibration --output artifacts/dino-pca.npz
docker compose exec app python inspect_defects.py evaluate --model artifacts/dino-pca.npz --manifest data/demo/test.json --output artifacts/report.json
```
