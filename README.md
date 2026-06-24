# ai-indonesian-fritters-counter

Repository for **training the YOLO model** that detects gorengan (Indonesian fritters) stock status. This repo is **used only during training** (creating / retraining the model) and is not run in production.

## Model Classes

`nc=4` → `['kosong', 'meja', 'penuh', 'sedikit']` — detects the fill status of the gorengan table/tray.

## Dataset

Roboflow YOLO export lives in `gorengan-conter.yolo26/`:

| Split | Images |
|-------|-------:|
| train | 163 |
| valid | 16 |
| test  | 13 |

> The dataset (`*.zip` + the `gorengan-conter.yolo26/` folder) and model weights (`*.pt`) are **not committed** to git (see `.gitignore`). Dataset source: Roboflow workspace `naufalfirdaus`.

## Setup (Python + uv)

```bash
# 1. Install uv → https://docs.astral.sh/uv/
# 2. Create the environment & install dependencies (ultralytics, etc.)
uv sync

# 3. (Optional) verify the environment
uv run python -c "import ultralytics; print(ultralytics.__version__)"
```

## Training (example)

```bash
uv run yolo detect train \
  data=gorengan-conter.yolo26/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640
```

Training results are saved to `runs/detect/train/`. The best model is `runs/detect/train/weights/best.pt`.

## After Training

Copy `best.pt` to the backend repo (`backend-ssb-ai/models/best.pt`) for inference. Also record the dataset version, hyperparameters, and metrics (mAP) so the run is reproducible when retraining.

> Note: the Setup & Training steps above describe the intended workflow. The project is not scaffolded yet (`pyproject.toml` does not exist), so `uv sync` will only work after dependencies are set up.
