# ai-indonesian-fritters-counter

Repository for **training the YOLO model** that detects gorengan (Indonesian fritters) stock status. This repo is **used only during training** (creating / retraining the model) and is not run in production.

## Model Classes

`nc=5` → `['habis', 'hampir habis', 'meja', 'penuh', 'sedikit']` — detects the fill status of each tray on the gorengan table. The fill state is bucketed by volume:

| Index | Class | Meaning |
|------:|-------|---------|
| 0 | `habis` | empty (0%) |
| 1 | `hampir habis` | almost empty (1–25%) |
| 2 | `meja` | the table (1 per frame; region-of-interest anchor) |
| 3 | `penuh` | full (51–100%) |
| 4 | `sedikit` | few (26–50%) |

> **V2 replaced the V1 scheme** (`nc=4` = `kosong/meja/penuh/sedikit`). V2 splits the empty→low range into two finer buckets (`habis` + `hampir habis`) and drops `kosong`. The class **indices changed**, so a V1 `best.pt` is not interchangeable with V2 labels — the backend's class mapping must be updated when the V2 model is deployed.

## Dataset

Roboflow YOLO (segmentation) export lives in `v2-gorengan-counter.yolo26/`:

| Split | Images |
|-------|-------:|
| train | 169 |
| valid | 12 |
| test  | 10 |

> The dataset (`*.zip` + the `v2-gorengan-counter.yolo26/` folder) and model weights (`*.pt`) are **not committed** to git (see `.gitignore`). Dataset source: Roboflow workspace `naufalfirdaus`, project `v2-gorengan-counter`.

## Setup (Python + uv)

```bash
# 1. Install uv → https://docs.astral.sh/uv/
# 2. Create the environment & install dependencies (ultralytics, etc.)
uv sync

# 3. (Optional) verify the environment
uv run python -c "import ultralytics; print(ultralytics.__version__)"
```

## Training

The Roboflow export ships **segmentation polygons**, but we train a **detection** model, so
there are two steps (see `TRAINING.md` for the full rationale):

```bash
# 1. Convert polygons → axis-aligned boxes (writes dataset_det/, originals untouched)
uv run python scripts/seg_to_det.py

# 2. Train from config.yaml (reads dataset_det/data.yaml, enforces GPU)
uv run python scripts/train.py
```

Hyperparameters live in `config.yaml`. Results are saved to `runs/<name>/`; the best model is
`runs/<name>/weights/best.pt`.

## After Training

Copy `best.pt` to the backend repo (`backend-ssb-ai/models/best.pt`) for inference. Also record the
dataset version, hyperparameters, and metrics (mAP) so the run is reproducible when retraining.
