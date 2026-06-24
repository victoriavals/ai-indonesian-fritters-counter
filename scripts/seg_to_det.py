"""Convert the YOLO *segmentation* dataset (polygons) into a *detection*
dataset (axis-aligned bounding boxes).

The Roboflow export `gorengan-conter.yolo26/` stores each object as a polygon:

    <class> x1 y1 x2 y2 x3 y3 ...        (normalized 0..1)

For detection training we need axis-aligned boxes instead:

    <class> cx cy w h                    (normalized 0..1)

This script reads every label, takes the min/max of each polygon's points to
form a tight box, and writes a brand-new dataset at `dataset_det/`. The original
dataset is never modified. Images are hard-linked (no extra disk usage) when
possible, otherwise copied.

Usage:
    uv run python scripts/seg_to_det.py
    uv run python scripts/seg_to_det.py --src gorengan-conter.yolo26 --dst dataset_det
    uv run python scripts/seg_to_det.py --min-size 0.002   # drop tinier boxes
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Repo root = parent of this script's `scripts/` folder.
REPO_ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["kosong", "meja", "penuh", "sedikit"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def polygon_to_box(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Turn a flat list of normalized polygon points into (cx, cy, w, h).

    Returns None if the polygon is malformed (odd number of values or empty).
    Coordinates are clamped to the [0, 1] range.
    """
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None

    xs = [min(max(c, 0.0), 1.0) for c in coords[0::2]]
    ys = [min(max(c, 0.0), 1.0) for c in coords[1::2]]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def convert_label_file(src_file: Path, dst_file: Path, min_size: float) -> tuple[dict[int, int], int]:
    """Convert one label file. Returns (per-class box counts, dropped count)."""
    per_class: dict[int, int] = {}
    dropped = 0

    lines_out: list[str] = []
    for raw in src_file.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        coords = [float(p) for p in parts[1:]]

        box = polygon_to_box(coords)
        if box is None:
            dropped += 1
            continue

        cx, cy, w, h = box
        # Drop boxes that are essentially zero-area (e.g. degenerate 2-point polygons).
        if w < min_size or h < min_size:
            dropped += 1
            continue

        lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        per_class[cls] = per_class.get(cls, 0) + 1

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return per_class, dropped


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link an image (cheap) or fall back to copying."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_data_yaml(dst_root: Path) -> None:
    """Write a detection-ready data.yaml with an absolute root path."""
    names = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    content = (
        f"path: {dst_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    (dst_root / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation labels to detection boxes.")
    parser.add_argument("--src", default="gorengan-conter.yolo26", help="source dataset dir (relative to repo root)")
    parser.add_argument("--dst", default="dataset_det", help="output dataset dir (relative to repo root)")
    parser.add_argument("--min-size", type=float, default=0.001, help="drop boxes smaller than this (normalized)")
    args = parser.parse_args()

    src_root = (REPO_ROOT / args.src).resolve()
    dst_root = (REPO_ROOT / args.dst).resolve()

    if not src_root.exists():
        raise SystemExit(f"Source dataset not found: {src_root}")

    totals: dict[int, int] = {}
    total_dropped = 0
    total_images = 0

    for split in SPLITS:
        src_labels = src_root / split / "labels"
        src_images = src_root / split / "images"
        if not src_labels.exists():
            print(f"[skip] {split}: no labels folder")
            continue

        split_class: dict[int, int] = {}
        split_dropped = 0
        split_images = 0

        for label_file in sorted(src_labels.glob("*.txt")):
            dst_label = dst_root / split / "labels" / label_file.name
            per_class, dropped = convert_label_file(label_file, dst_label, args.min_size)
            for cls, n in per_class.items():
                split_class[cls] = split_class.get(cls, 0) + n
                totals[cls] = totals.get(cls, 0) + n
            split_dropped += dropped
            total_dropped += dropped

            # Link the matching image (try each known extension).
            for ext in IMAGE_EXTS:
                img = src_images / (label_file.stem + ext)
                if img.exists():
                    link_or_copy(img, dst_root / split / "images" / img.name)
                    split_images += 1
                    total_images += 1
                    break

        counts = ", ".join(f"{CLASS_NAMES[c]}={split_class.get(c, 0)}" for c in range(len(CLASS_NAMES)))
        print(f"[{split:5}] images={split_images:4}  dropped={split_dropped:3}  | {counts}")

    write_data_yaml(dst_root)

    print("\n=== TOTAL ===")
    for c in range(len(CLASS_NAMES)):
        print(f"  {CLASS_NAMES[c]:8}: {totals.get(c, 0)} boxes")
    print(f"  dropped (degenerate): {total_dropped}")
    print(f"  images linked       : {total_images}")
    print(f"\nDetection dataset ready at: {dst_root}")
    print(f"data.yaml: {(dst_root / 'data.yaml')}")


if __name__ == "__main__":
    main()
