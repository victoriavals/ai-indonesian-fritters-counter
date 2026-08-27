"""Validasi dataset deteksi dan hasilkan statistik untuk Tabel T2 paper.

KENAPA SCRIPT INI ADA
---------------------
Dua alasan konkret:

1. Total instance per kelas untuk `hampir habis` dan `sedikit` TIDAK tercatat di registry
   mana pun (MODELS.md hanya mencatat total `penuh` 5.135, `habis` 269, dan jumlah per-split
   test). Paper membutuhkan tabel distribusi lengkap. Script ini menghitungnya.

2. Klaim "0 kotak degenerate" dan "tepat satu `meja` per citra" dipakai di metodologi paper,
   tetapi belum pernah diverifikasi ulang pada split bersih. Script ini membuktikannya.

CATATAN PENTING SOAL CARA MENGHITUNG
------------------------------------
Berkas label Roboflow sering TIDAK diakhiri newline. Karena itu menghitung kelas dengan
`cat */labels/*.txt | awk ...` MENGGABUNGKAN baris terakhir satu berkas dengan baris pertama
berkas berikutnya - insiden nyata di project ini pernah melaporkan `meja` sebagai 6 padahal
sebenarnya ~169. Script ini membaca SETIAP BERKAS SECARA TERPISAH. Jangan "optimasi" ini
menjadi satu pipe.

INPUT
-----
Direktori dataset deteksi berformat YOLO: <root>/{train,valid,test}/{images,labels}
Default: dataset_det_clean (split group-aware, 501/80/80)

OUTPUT
------
- research/results/dataset_statistics_<root>.csv  -> T2 (instance per kelas per split)
- research/results/dataset_validation_<root>.csv  -> temuan per berkas (kosong = bersih)
- Ringkasan ke stdout

CARA MENJALANKAN
----------------
    uv run python research/validate_dataset.py
    uv run python research/validate_dataset.py --root dataset_det       # split acak, pembanding
    uv run python research/validate_dataset.py --check-images           # buka tiap citra (lambat)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "research" / "results"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Urutan kelas WAJIB sama dengan data.yaml. Diambil dari data.yaml bila ada, dengan
# daftar ini sebagai cadangan - urutan yang salah menukar label kelas tanpa menimbulkan error.
FALLBACK_CLASS_NAMES = ["habis", "hampir habis", "meja", "penuh", "sedikit"]
ANCHOR_CLASS = "meja"


def load_class_names(root: Path) -> list[str]:
    """Baca `names:` dari data.yaml. Jatuh ke daftar cadangan bila tidak terbaca."""
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        print(f"[warn] {data_yaml} tidak ada - memakai urutan kelas cadangan")
        return list(FALLBACK_CLASS_NAMES)
    try:
        import yaml  # tersedia via ultralytics

        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = cfg.get("names")
        if isinstance(names, dict):  # format {0: 'a', 1: 'b'}
            names = [names[k] for k in sorted(names)]
        if not isinstance(names, list) or not names:
            raise ValueError("kunci 'names' bukan daftar tak-kosong")
        return [str(n) for n in names]
    except Exception as exc:  # noqa: BLE001 - laporkan, jangan gagal total
        print(f"[warn] gagal membaca names dari {data_yaml} ({exc}) - memakai cadangan")
        return list(FALLBACK_CLASS_NAMES)


def parse_label_file(
    path: Path, n_classes: int
) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Baca satu berkas label YOLO. Mengembalikan (daftar kotak, daftar temuan)."""
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return boxes, [f"tidak terbaca: {exc}"]

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"baris {lineno}: {len(parts)} kolom (harus 5)")
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(f"baris {lineno}: angka tidak sah")
            continue

        if not 0 <= cid < n_classes:
            issues.append(f"baris {lineno}: class id {cid} di luar rentang 0..{n_classes - 1}")
            continue
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not 0.0 <= val <= 1.0:
                issues.append(f"baris {lineno}: {name}={val:.6f} di luar [0,1]")
        if w <= 0 or h <= 0:
            issues.append(f"baris {lineno}: kotak degenerate w={w:.6f} h={h:.6f}")
        boxes.append((cid, cx, cy, w, h))
    return boxes, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validasi dataset deteksi YOLO + statistik T2.")
    ap.add_argument("--root", default="dataset_det_clean", help="direktori dataset, relatif ke repo root")
    ap.add_argument("--check-images", action="store_true", help="buka setiap citra untuk cek korupsi (lambat)")
    args = ap.parse_args()

    root = (REPO_ROOT / args.root).resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: dataset tidak ditemukan: {root}")

    class_names = load_class_names(root)
    n_classes = len(class_names)

    stamp = "(tanpa stempel)"
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        lines = data_yaml.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip().startswith("#"):
            stamp = lines[0].strip()

    print(f"Dataset   : {root}")
    print(f"Provenance: {stamp}")
    print(f"Kelas     : {n_classes} -> {class_names}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, list[int]] = {s: [0] * n_classes for s in SPLITS}
    n_images: dict[str, int] = {s: 0 for s in SPLITS}
    n_labels: dict[str, int] = {s: 0 for s in SPLITS}
    findings: list[dict[str, str]] = []
    anchor_hist: dict[int, int] = {}
    empty_labels = 0

    anchor_idx = class_names.index(ANCHOR_CLASS) if ANCHOR_CLASS in class_names else None

    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not lbl_dir.is_dir():
            print(f"[skip] {split}: tidak ada folder labels")
            continue

        images = (
            sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if img_dir.is_dir()
            else []
        )
        labels = sorted(lbl_dir.glob("*.txt"))
        n_images[split], n_labels[split] = len(images), len(labels)

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        for stem in sorted(img_stems - lbl_stems):
            findings.append({"split": split, "berkas": stem, "masalah": "citra tanpa berkas label"})
        for stem in sorted(lbl_stems - img_stems):
            findings.append({"split": split, "berkas": stem, "masalah": "label tanpa berkas citra"})

        for lbl in labels:
            boxes, issues = parse_label_file(lbl, n_classes)
            for issue in issues:
                findings.append({"split": split, "berkas": lbl.name, "masalah": issue})
            if not boxes:
                empty_labels += 1
                findings.append({"split": split, "berkas": lbl.name, "masalah": "label kosong (0 kotak)"})
            for cid, *_rest in boxes:
                counts[split][cid] += 1
            if anchor_idx is not None:
                n_anchor = sum(1 for cid, *_r in boxes if cid == anchor_idx)
                anchor_hist[n_anchor] = anchor_hist.get(n_anchor, 0) + 1

        if args.check_images:
            try:
                from PIL import Image

                for img in images:
                    try:
                        with Image.open(img) as im:
                            im.verify()
                    except Exception as exc:  # noqa: BLE001
                        findings.append(
                            {"split": split, "berkas": img.name, "masalah": f"citra rusak: {exc}"}
                        )
            except ImportError:
                print("[warn] Pillow tidak tersedia - --check-images dilewati")

    # ---------- Tabel T2 ----------
    header = f"{'kelas':16s}" + "".join(f"{s:>9s}" for s in SPLITS) + f"{'TOTAL':>9s}{'%':>8s}"
    print(header)
    print("-" * len(header))
    grand = sum(sum(counts[s]) for s in SPLITS)
    rows: list[dict[str, str]] = []
    for i, cname in enumerate(class_names):
        per = [counts[s][i] for s in SPLITS]
        tot = sum(per)
        pct = (tot / grand * 100) if grand else 0.0
        print(f"{cname:16s}" + "".join(f"{v:9d}" for v in per) + f"{tot:9d}{pct:7.1f}%")
        row = {"kelas": cname, "total": str(tot), "persen": f"{pct:.2f}"}
        row.update({s: str(counts[s][i]) for s in SPLITS})
        rows.append(row)
    print("-" * len(header))
    print(f"{'instance':16s}" + "".join(f"{sum(counts[s]):9d}" for s in SPLITS) + f"{grand:9d}")
    print(f"{'citra':16s}" + "".join(f"{n_images[s]:9d}" for s in SPLITS) + f"{sum(n_images.values()):9d}")
    print(f"{'label':16s}" + "".join(f"{n_labels[s]:9d}" for s in SPLITS) + f"{sum(n_labels.values()):9d}")

    # Ketimpangan kelas - angka yang harus dibahas jujur di paper.
    totals = {class_names[i]: sum(counts[s][i] for s in SPLITS) for i in range(n_classes)}
    nonzero = {k: v for k, v in totals.items() if v > 0}
    if nonzero:
        kmax, vmax = max(nonzero.items(), key=lambda kv: kv[1])
        kmin, vmin = min(nonzero.items(), key=lambda kv: kv[1])
        print(f"\nKetimpangan kelas: {kmax} {vmax} : {kmin} {vmin}  ->  {vmax / vmin:.1f}:1")

    if anchor_idx is not None:
        print(f"\nInstance '{ANCHOR_CLASS}' per citra: {dict(sorted(anchor_hist.items()))}")
        if set(anchor_hist) == {1}:
            print(f"  OK: tepat satu '{ANCHOR_CLASS}' di setiap citra"
                  " (asumsi anchor pipeline pemetaan slot terpenuhi)")
        else:
            print(f"  PERHATIAN: tidak semua citra punya tepat satu '{ANCHOR_CLASS}'."
                  " Pemetaan slot mengandalkan asumsi ini.")

    # ---------- Tulis CSV ----------
    stats_csv = RESULTS_DIR / f"dataset_statistics_{root.name}.csv"
    val_csv = RESULTS_DIR / f"dataset_validation_{root.name}.csv"
    try:
        with open(stats_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["kelas", *SPLITS, "total", "persen"])
            w.writeheader()
            w.writerows(rows)
        with open(val_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["split", "berkas", "masalah"])
            w.writeheader()
            w.writerows(findings)
    except OSError as exc:
        sys.exit(f"ERROR: gagal menulis CSV: {exc}")

    print(f"\nStatistik (T2) : {stats_csv}")
    print(f"Temuan validasi: {val_csv}")

    if findings:
        print(f"\n{len(findings)} TEMUAN (label kosong: {empty_labels}). Contoh 10 pertama:")
        for f in findings[:10]:
            print(f"  [{f['split']}] {f['berkas']}: {f['masalah']}")
        print("\nCatatan: label kosong belum tentu salah (citra tanpa objek), tetapi pada dataset"
              " ini setiap citra seharusnya punya minimal satu kotak 'meja'.")
    else:
        print("\nBERSIH: tidak ada label rusak, kotak degenerate, atau berkas tanpa pasangan.")


if __name__ == "__main__":
    main()
