"""Evaluate a trained checkpoint on a dataset split, and write the result next to the run.

Usage:
    # evaluate a run's own best.pt on its own test split
    uv run python scripts/evaluate.py --weights runs/yolo26s_det_v21_clean/weights/best.pt \
                                      --data dataset_det_clean/data.yaml --split test

    # head-to-head: same data, two checkpoints (repeat the command, vary --weights/--name)
    uv run python scripts/evaluate.py -w ../backend-ssb-ai/models/best.pt \
                                      -d dataset_fair_eval/data.yaml --name fair_live

    # machine-readable output for scripting
    uv run python scripts/evaluate.py -w runs/x/weights/best.pt -d dataset_det/data.yaml --json

WHY THIS SCRIPT EXISTS — two traps it closes
--------------------------------------------
1. **Ultralytics' global `runs_dir` hijacks relative `project` paths.** On this machine it is set
   to a *different* project (`D:\\computer-vision\\density-aware-yolo26-vehicle-counting\\runs`),
   so calling `model.val(project="runs", ...)` silently writes plots into that other repo. It
   happened on 2026-07-31 and the output had to be moved back by hand. `train.py` already guards
   against this by forcing an absolute `project`; this script does the same for evaluation, which
   is repeated after every retrain and so is just as exposed.

2. **`workers>0` deadlocks the dataloader on Windows.** Ultralytics defaults `val()` to 8 workers.
   A `val()` call left at the default hung for 10 minutes and had to be killed; `workers=0` runs
   it in seconds. `config.yaml` carries the same warning for training. Hard-coded to 0 here, with
   `--workers` available only if you deliberately want to override it.

Everything is printed AND returned as JSON (`--json`), so a comparison table can be assembled
without re-parsing Ultralytics' console output — which is ANSI-coloured and uses `\\r` progress
redraws, and whose column order is easy to misread (P, R, mAP50, mAP50-95 are fields 4-7, not 3-6;
misreading them produced a bogus "mAP50-95 > mAP50" report earlier the same day).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    """Resolve a path relative to the repo root (so the script works from any cwd)."""
    q = Path(p)
    return q if q.is_absolute() else (REPO_ROOT / q).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    ap.add_argument("-w", "--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("-d", "--data", required=True, help="path to a data.yaml")
    ap.add_argument("-s", "--split", default="test", choices=("train", "val", "test"),
                    help="which split to evaluate (default: test)")
    ap.add_argument("-n", "--name", default=None,
                    help="output folder name under runs/ (default: <weights-run>_<split>)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (match training; default 640)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers. KEEP 0 ON WINDOWS — higher values deadlock.")
    ap.add_argument("--no-plots", action="store_true", help="skip curve/confusion-matrix images")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="print a JSON blob instead of a table (for scripting)")
    args = ap.parse_args()

    weights = resolve(args.weights)
    data = resolve(args.data)
    if not weights.is_file():
        sys.exit(f"ERROR: weights not found: {weights}")
    if not data.is_file():
        sys.exit(f"ERROR: data.yaml not found: {data}")

    # Default run name: the training run the weights came from, plus the split.
    # runs/<run>/weights/best.pt -> "<run>_test"
    if args.name:
        name = args.name
    else:
        parts = weights.parts
        stem = parts[-3] if len(parts) >= 3 and parts[-2] == "weights" else weights.stem
        name = f"{stem}_{args.split}"

    # THE important line: an ABSOLUTE project dir, so the global runs_dir cannot redirect us.
    project = (REPO_ROOT / "runs").resolve()

    from ultralytics import YOLO  # imported late so --help stays instant

    if not args.as_json:
        print(f"[WEIGHTS] {weights}")
        print(f"[DATA   ] {data}  (split={args.split})")
        print(f"[OUTPUT ] {project / name}")

    model = YOLO(str(weights))

    def run_val():
        return model.val(
            data=str(data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=str(project),
            name=name,
            plots=not args.no_plots,
            verbose=not args.as_json,
        )

    if args.as_json:
        # Ultralytics prints its banner/summary regardless of verbose=False, which would
        # corrupt the JSON blob for a caller doing `evaluate.py --json | jq`.
        #
        # redirect_stdout ALONE IS NOT ENOUGH: ultralytics' LOGGER already captured the
        # original sys.stdout object when its StreamHandler was built, so rebinding
        # sys.stdout leaves the logger writing to the real terminal. Both have to be
        # muted — the handler stream *and* plain print() calls.
        import contextlib
        import io
        import logging
        import os

        from ultralytics.utils import LOGGER

        devnull = open(os.devnull, "w", encoding="utf-8")
        saved = [(h, h.stream) for h in LOGGER.handlers if isinstance(h, logging.StreamHandler)]
        try:
            for h, _ in saved:
                h.setStream(devnull)
            with contextlib.redirect_stdout(io.StringIO()):
                r = run_val()
        finally:
            for h, stream in saved:
                h.setStream(stream)
            devnull.close()
    else:
        r = run_val()

    names = model.names  # {idx: name} — authoritative, avoids hard-coding class order
    per_class = {}
    for i, cls in enumerate(r.box.ap_class_index):
        per_class[names[int(cls)]] = {
            "precision": round(float(r.box.p[i]), 4),
            "recall": round(float(r.box.r[i]), 4),
            "mAP50": round(float(r.box.ap50[i]), 4),
            "mAP50-95": round(float(r.box.ap[i]), 4),
        }

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "all": {
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
        },
        "per_class": per_class,
    }

    # Persist beside the eval output so a number can always be traced to its run.
    save_dir = Path(r.save_dir) if getattr(r, "save_dir", None) else (project / name)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "eval_meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.as_json:
        print(json.dumps(out, indent=2))
        return

    print(f"\n{'class':<14} {'P':>7} {'R':>7} {'mAP50':>8} {'mAP50-95':>9}")
    a = out["all"]
    print(f"{'all':<14} {a['precision']:>7.3f} {a['recall']:>7.3f} {a['mAP50']:>8.4f} {a['mAP50-95']:>9.4f}")
    for cls, m in per_class.items():
        print(f"{cls:<14} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['mAP50']:>8.4f} {m['mAP50-95']:>9.4f}")

    # mAP50-95 averages over stricter IoU thresholds, so it can never exceed mAP50.
    # If it does, the numbers were misread — fail loudly rather than report nonsense.
    if a["mAP50-95"] > a["mAP50"]:
        sys.exit("\nERROR: mAP50-95 > mAP50 — impossible; metrics were misread.")

    print(f"\nSaved: {save_dir / 'eval_meta.json'}")


if __name__ == "__main__":
    main()
