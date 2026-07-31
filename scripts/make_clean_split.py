"""Rebuild a Roboflow export's train/valid/test split so near-duplicate frames
never straddle two splits.

WHY THIS EXISTS
---------------
The V2.1 export (`v2.1-gorengan-counter.yolo26`) was split **randomly over frames**.
But the frames are screenshots grabbed minutes apart from the same CCTV recording, so
a random split scatters near-identical images across train/valid/test. Measured on the
as-shipped V2.1 split: 36% of test images had a train twin at 64x64 grayscale pixel-MAE
< 5 (a random-pair baseline is ~38), and the closest pair was 1 min 54 s apart in the
same scene. Any metric computed on that split is optimistic — the model is scored on
near-copies of what it memorised.

WHAT THIS DOES
--------------
1. Pools every image from all three source splits.
2. Groups them by visual similarity: an edge between two frames whose thumbnail MAE is
   below `--threshold`, then connected components (union-find). One component ~= one
   "scene" / capture session.
3. Assigns **whole components** to train/valid/test, so no component is ever split.
   Components are placed largest-first into whichever split is furthest below its quota.
4. Writes a new dataset directory in the SAME segmentation format, with images
   hard-linked (no extra disk) and label files copied verbatim.
5. Reports the resulting per-split class balance and re-measures cross-split leakage so
   the improvement is verified, not assumed.

Feed the output to `seg_to_det.py` as usual:

    uv run python scripts/make_clean_split.py
    uv run python scripts/seg_to_det.py --src v2.1-clean-split.yolo26 --dst dataset_det_clean

Then point `config.yaml`'s `data:` at `dataset_det_clean/data.yaml`.

NOTE ON THE THRESHOLD
---------------------
`--threshold` is in mean-absolute-difference units on 64x64 grayscale thumbnails (0-255).
Higher = more aggressive grouping = fewer, larger components = a stricter (and smaller
effective) test set. 12 is a reasonable default here: well above the near-duplicate range
(<5) yet well below the random-pair baseline (~38). Inspect the reported component sizes;
if one component swallows most of the dataset, lower it.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES = ["habis", "hampir habis", "meja", "penuh", "sedikit"]


# --------------------------------------------------------------------------- utils
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collect_images(src_root: Path) -> list[Path]:
    """Every image across all source splits, sorted for determinism."""
    out: list[Path] = []
    for split in SPLITS:
        d = src_root / split / "images"
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    return out


def thumbnails(paths: list[Path], size: int) -> np.ndarray:
    """(N, size*size) float32 grayscale thumbnails."""
    rows = np.empty((len(paths), size * size), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((size, size))
        rows[i] = np.asarray(im, dtype=np.float32).ravel()
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(paths)} thumbnail")
    return rows


def pairwise_mae(t: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Full (N, N) mean-absolute-difference matrix, computed in row chunks to bound RAM."""
    n = t.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # |a - b| averaged over pixels, broadcast one chunk of rows against all rows.
        out[start:stop] = np.abs(t[start:stop, None, :] - t[None, :, :]).mean(axis=2)
    return out


def label_for(img: Path, src_root: Path) -> Path | None:
    """The label file matching an image, searched across all source splits (the image may
    move between splits, but its label lives next to wherever it came from)."""
    for split in SPLITS:
        cand = src_root / split / "labels" / (img.stem + ".txt")
        if cand.is_file():
            return cand
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def class_counts(label_files: list[Path]) -> np.ndarray:
    c = np.zeros(len(CLASS_NAMES), dtype=int)
    for lf in label_files:
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                idx = int(float(parts[0]))
                if 0 <= idx < len(c):
                    c[idx] += 1
    return c


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Group-aware re-split of a Roboflow export.")
    ap.add_argument("--src", default="v2.1-gorengan-counter.yolo26", help="source export (relative to repo root)")
    ap.add_argument("--dst", default="v2.1-clean-split.yolo26", help="output dataset dir (relative to repo root)")
    ap.add_argument("--threshold", type=float, default=12.0, help="thumbnail MAE below which two frames are 'the same scene'")
    ap.add_argument("--thumb", type=int, default=64, help="thumbnail edge in px")
    ap.add_argument("--val-frac", type=float, default=0.12, help="target fraction of images for valid")
    ap.add_argument("--test-frac", type=float, default=0.12, help="target fraction of images for test")
    args = ap.parse_args()

    src_root = (REPO_ROOT / args.src).resolve()
    dst_root = (REPO_ROOT / args.dst).resolve()
    if not src_root.is_dir():
        raise SystemExit(f"Source dataset not found: {src_root}")
    if dst_root.exists():
        # Never merge into a stale directory — that is the bug seg_to_det.py has.
        print(f"[clean] removing existing {dst_root}")
        shutil.rmtree(dst_root)

    images = collect_images(src_root)
    if not images:
        raise SystemExit(f"No images found under {src_root}")
    print(f"Pooled {len(images)} images from {args.src}")

    print("Computing thumbnails...")
    t = thumbnails(images, args.thumb)

    print("Computing pairwise similarity...")
    d = pairwise_mae(t)

    # --- group near-duplicates into components -----------------------------------
    uf = UnionFind(len(images))
    iu = np.triu_indices(len(images), k=1)
    close = np.nonzero(d[iu] < args.threshold)[0]
    for k in close:
        uf.union(int(iu[0][k]), int(iu[1][k]))

    groups: dict[int, list[int]] = {}
    for i in range(len(images)):
        groups.setdefault(uf.find(i), []).append(i)
    comps = sorted(groups.values(), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    print(f"\n{len(comps)} scene-groups at MAE < {args.threshold}")
    print(f"  sizes: largest={sizes[0]}, median={int(np.median(sizes))}, singletons={sum(1 for s in sizes if s == 1)}")
    if sizes[0] > 0.5 * len(images):
        print("  WARNING: one group holds >50% of the dataset — lower --threshold for a usable split.")

    # --- assign whole groups to splits -------------------------------------------
    n = len(images)
    quota = {
        "valid": args.val_frac * n,
        "test": args.test_frac * n,
        "train": (1.0 - args.val_frac - args.test_frac) * n,
    }
    assigned: dict[str, list[int]] = {s: [] for s in SPLITS}
    # Largest group first, into whichever split is furthest below quota (as a fraction of
    # its own quota) — keeps small splits from being starved or overshot by one big group.
    for comp in comps:
        target = max(assigned, key=lambda s: (quota[s] - len(assigned[s])) / quota[s])
        assigned[target].extend(comp)

    # --- write the new dataset ----------------------------------------------------
    print()
    missing_labels = 0
    for split in SPLITS:
        idxs = assigned[split]
        for i in idxs:
            img = images[i]
            link_or_copy(img, dst_root / split / "images" / img.name)
            lf = label_for(img, src_root)
            if lf is None:
                missing_labels += 1
                continue
            dst_lf = dst_root / split / "labels" / lf.name
            dst_lf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lf, dst_lf)

    names = ", ".join(f"'{x}'" for x in CLASS_NAMES)
    (dst_root / "data.yaml").write_text(
        f"# regrouped_from: {args.src} (group-aware split, MAE threshold {args.threshold})\n"
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )

    # --- report -------------------------------------------------------------------
    print(f"{'split':6} {'images':>7} {'groups':>7}  " + "  ".join(f"{c:>13}" for c in CLASS_NAMES))
    for split in SPLITS:
        idxs = set(assigned[split])
        ngroups = sum(1 for c in comps if idxs.issuperset(c))
        lfs = sorted((dst_root / split / "labels").glob("*.txt"))
        cc = class_counts(lfs)
        print(f"{split:6} {len(assigned[split]):7} {ngroups:7}  " + "  ".join(f"{v:13}" for v in cc))
    if missing_labels:
        print(f"  WARNING: {missing_labels} images had no matching label file")

    # --- verify the leakage actually went away ------------------------------------
    print("\nCross-split leakage check (nearest train neighbour, thumbnail MAE):")
    tr = np.array(assigned["train"])
    for split in ("valid", "test"):
        ev = np.array(assigned[split])
        if ev.size == 0 or tr.size == 0:
            continue
        sub = d[np.ix_(ev, tr)]
        mins = sub.min(axis=1)
        under5 = int((mins < 5).sum())
        print(f"  {split:5}: min={mins.min():5.2f}  median={np.median(mins):5.2f}  "
              f"| MAE<5: {under5}/{len(mins)} ({under5 / len(mins) * 100:.1f}%)")

    print(f"\nClean split written to: {dst_root}")
    print(f"Next: uv run python scripts/seg_to_det.py --src {args.dst} --dst dataset_det_clean")


if __name__ == "__main__":
    main()
