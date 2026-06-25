"""Train the gorengan detection model (YOLO26) from `config.yaml`.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config config.yaml

What this script does:
- reads hyperparameters from `config.yaml`,
- **enforces CUDA/GPU training** (fails fast if no GPU is found, since CPU
  training would be far too slow) — pass `--cpu` only to deliberately override,
- resolves the dataset path relative to the repo root,
- runs Ultralytics training,
- writes a `run_meta.json` (model + hyperparameters + final metrics) into the
  run folder, so each training run is reproducible (see project conventions).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gorengan detection model.")
    parser.add_argument("--config", default="config.yaml", help="path to config (relative to repo root)")
    parser.add_argument("--cpu", action="store_true", help="allow CPU training (NOT recommended — very slow)")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (e.g. for a smoke test)")
    parser.add_argument("--name", default=None, help="override run name (e.g. 'smoke')")
    args = parser.parse_args()

    cfg_path = (REPO_ROOT / args.config).resolve()
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found: {cfg_path}")
    cfg = load_config(cfg_path)

    # CLI overrides (handy for quick smoke tests without editing config.yaml)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.name is not None:
        cfg["name"] = args.name

    # --- Enforce GPU unless explicitly overridden ---
    if torch.cuda.is_available():
        print(f"[GPU] Training on: {torch.cuda.get_device_name(0)} "
              f"({round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB)")
    else:
        if not args.cpu:
            sys.exit(
                "ERROR: No CUDA GPU detected. Training on CPU would be far too slow.\n"
                "Fix the CUDA/torch install, or pass --cpu to override deliberately."
            )
        print("[WARN] CUDA not available — training on CPU because --cpu was passed.")
        cfg["device"] = "cpu"

    # --- Resolve dataset path relative to the repo root ---
    data = cfg.get("data")
    if data and not Path(data).is_absolute():
        cfg["data"] = str((REPO_ROOT / data).resolve())

    # --- Force the output dir into THIS repo (absolute) ---
    # Ultralytics has a global settings runs_dir that may point elsewhere
    # (e.g. another project on a full C: drive). Making `project` absolute
    # guarantees runs land in this repo on D:, without touching global settings.
    project = cfg.get("project", "runs")
    if not Path(project).is_absolute():
        cfg["project"] = str((REPO_ROOT / project).resolve())

    model_name = cfg.pop("model")
    print(f"[MODEL] {model_name}")
    print(f"[DATA ] {cfg['data']}")

    model = YOLO(model_name)
    results = model.train(**cfg)

    # --- Write run metadata for reproducibility ---
    save_dir = Path(model.trainer.save_dir)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "dataset": cfg.get("data"),
        "config": cfg,
    }
    try:
        meta["metrics"] = results.results_dict
    except Exception:
        meta["metrics"] = None
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n[DONE] Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"[DONE] Run metadata: {save_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
