# Training Pipeline — Gorengan Counter

**Goal:** for each frame, count the containers on the table by fill state — `kosong` (empty), `sedikit` (few), `penuh` (full).

## Decision: Detection (not Segmentation)

We train a **detection** model (`yolo26s`), even though the Roboflow dataset ships as **segmentation polygons**.

Reasoning:
- Counting objects only needs **class + location**, not pixel masks. The count is simply the number of boxes per class.
- Detection is **lighter and faster** to train on an 8 GB GPU and to serve in the backend.
- No annotation is wasted: polygons are **converted to axis-aligned bounding boxes** in a separate dataset copy (the original stays untouched).
- `meja` (the table, 1 per frame) is kept (`nc=4`) as a free region-of-interest marker; the backend can use it to count only containers on the table.

## Environment

- GPU: NVIDIA RTX 4060 Ti, 8 GB VRAM
- Python 3.11, package manager: **uv**
- Framework: Ultralytics (YOLO26)

## Pipeline (step by step)

### Phase 1 — Project setup
- `uv init` the AI repo, add `ultralytics` + CUDA-enabled PyTorch, pin versions (`uv.lock`).
- Verify: GPU is visible to torch, and `yolo26s` weights are available.
- Output: reproducible environment.

### Phase 2 — Dataset preparation (polygon → box)
- Script `scripts/seg_to_det.py`: read each segmentation label, convert every polygon to an axis-aligned box (min/max of points), write a new dataset at `dataset_det/` (images hard-linked, originals untouched) + a matching `data.yaml`.
- Clean the ~38 degenerate 2-point polygons (drop or fix boxes that are empty/too small).
- Validate: box counts per class match expectations; a few converted samples render correctly.
- Output: `dataset_det/` ready for detection training.

### Phase 3 — Training config + script
- `config.yaml`: all hyperparameters in one place (reproducible).
- `train.py`: thin, documented wrapper around Ultralytics that reads `config.yaml`, sets the seed, trains, and writes a run-metadata file (dataset version, hyperparameters, metrics).
- Output: re-runnable training code.

### Phase 4 — Train
- `yolo26s.pt` pretrained (transfer learning), `imgsz=640`, `epochs=200`, `patience=50`, `batch=16` (fallback 8 / auto for 8 GB), `cache=True`, `close_mosaic=10`, `cos_lr=True`, `seed=0`.
- Output: `runs/detect/<name>/weights/best.pt`.

### Phase 5 — Evaluate
- Validate on the validation and **test** splits.
- Report **per-class** mAP50 / mAP50-95 (not just the global number — classes are imbalanced; `kosong` is weakest).
- Inspect the confusion matrix and a batch of predicted images visually.
- Output: metrics report + sanity-checked predictions.

### Phase 6 — Export + handoff
- Export `best.pt` (optionally ONNX for the backend).
- Record run metadata (which dataset version → which model + metrics).
- Copy `best.pt` to `backend-ssb-ai/models/best.pt`.

## Recommended starting hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| `model` | `yolo26s.pt` | small + pretrained (transfer learning) |
| `imgsz` | 640 | standard; dense scenes (~16 objects/frame) |
| `epochs` | 200 | small dataset → train long |
| `patience` | 50 | early stop to avoid overfitting |
| `batch` | 16 (→ 8 / auto) | fits 8 GB VRAM |
| `cache` | True | small dataset fits in RAM → faster |
| `close_mosaic` | 10 | disable mosaic near the end for clean convergence |
| `cos_lr` | True | smoother LR decay |
| `seed` | 0 | reproducibility |

## Notes / known data issues
- Dataset is YOLO **segmentation** polygons; meja ≈ 6–12 points, containers ≈ 4 points.
- Class instances (train): `sedikit` 1433, `penuh` 1082, `meja` 163, `kosong` 90 — `kosong` is the weakest class; watch its recall.
- ~38 degenerate 2-point polygons exist — handled in Phase 2.
- Validation split is small (16 images) → metrics are a bit noisy; trust the test split too.
