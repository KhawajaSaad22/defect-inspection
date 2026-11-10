# Verified progress

## Working

- DINOv2-small loaded from its official pretrained checkpoint and fit on four normal images.
- PCA and kNN baselines completed evaluation under clean, brightness, contrast, and blur shifts.
- Normal-only photometric augmentation completed as a paired ablation.
- Test manifests, calibration, content-overlap protection, model persistence, and CLI predictions work.
- Streamlit interface startup passed; image-upload interaction has not been automated end to end.
- Seven defect inspection tests and four active-selection tests passed.
- A demo defect image and learned-feature overlay are in artifacts/.

## Observed smoke results

Synthetic data: 8 training candidates, 20 calibration normals, 30 test images (15 defects),
four references, seed 42, 224-pixel inputs, DINOv2-small, CPU execution.

| Model | Clean FPR | Blur FPR | Contrast FPR | Defect recall across shifts |
|---|---:|---:|---:|---:|
| kNN | 0% | 6.7% | 0% | 100% |
| PCA | 0% | 20% | 6.7% | 100% |
| PCA + augmentation | 0% | 6.7% | 0% | 100% |

All DINO variants achieved image AUROC 1.0 on this deliberately easy toy data. This
does not establish industrial performance. In particular, one error changes the FPR
by 6.7 percentage points; the differences are not statistically persuasive. Reports
are saved in reports/ and experiments/smoke-v1/. Heatmap display calibration was later
improved without changing scoring or decisions; the primary dino-pca artifact was rebuilt.

## Next session

1. Obtain a licensed real MVTec AD category or collect normal/defect product photographs.
2. Prepare disjoint splits with prepare_mvtec.py; keep acquisition sessions separated.
3. Run at least three seeds and compare 1-shot and 4-shot models on several categories.
4. Install CUDA PyTorch in a dedicated environment and profile the RTX 4060 (8 GB).
5. Use real domain-shift sessions before making robustness claims.

No commercial dataset was downloaded, no claims of paper reproduction are made, and
no new repository was published. The sibling hybrid-rag environment was reused for
verification without modifying its dependencies; model weights were downloaded to the
standard Hugging Face cache. Scripts also support independent installation.
