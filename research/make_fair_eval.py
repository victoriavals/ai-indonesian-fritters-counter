"""Bangun ulang `dataset_fair_eval/` - himpunan uji adil untuk skenario S3.

KENAPA SCRIPT INI ADA
---------------------
`dataset_fair_eval/` (73 citra) SUDAH ADA di disk, tetapi TIDAK ADA script yang membuatnya.
MODELS.md:88 hanya menyebut "built by the snippet recorded in this repo's history". Artinya
tabel T7 - salah satu hasil terkuat paper (V2 mAP50 0,629 vs V2.1-clean 0,942) - bersandar
pada artefak yang tidak dapat direproduksi. Script ini menutup lubang itu.

ATURAN PENGECUALIAN - PERHATIKAN, DOKUMENTASI LAMA KELIRU
--------------------------------------------------------
MODELS.md:86 menulis: "the clean test split minus every frame V2 ever trained *or* validated on
(7 removed: 4445, 4491, 4492, 4501-4504)".

Diukur ulang, aturan itu TIDAK menghasilkan 7 frame:
  - clean_test  &  (V2 train + V2 valid)  = {4492, 4501, 4502, 4503, 4504}   -> 5 frame
  - clean_test  &  V2 test                = {4445, 4491}                     -> 2 frame
  - total                                                                    -> 7 frame

Jadi aturan yang BENAR-BENAR dipakai adalah: buang setiap frame clean-test yang muncul di
ekspor V2 SAMA SEKALI (train + valid + test), bukan hanya train+valid. Script ini memakai
aturan faktual tersebut supaya hasilnya identik dengan artefak yang sudah ada.

Aturan ini bersifat konservatif dan tetap dapat dipertahankan secara ilmiah: kedua model
sama-sama belum pernah melatih maupun memvalidasi pada 73 frame sisanya. Gunakan kalimat
"setiap frame yang muncul pada ekspor V2" di paper, JANGAN "yang pernah dilatih atau
divalidasi V2" - kalimat kedua tidak cocok dengan datanya.

IDENTITAS FRAME
---------------
Roboflow menamai berkas dengan hash konten (`Screenshot (4445)_png.rf.<hash>.png`), sehingga
hash untuk frame yang sama BISA berbeda antar ekspor. Karena itu pencocokan dilakukan pada
nomor screenshot `Screenshot (NNNN)`, yaitu identitas frame sumber - bukan pada nama berkas.

INPUT
-----
- dataset_det_clean/test        (80 citra, split group-aware)
- v2-gorengan-counter.yolo26/   (ekspor V2: 169/12/10) - hanya dibaca nama berkasnya

OUTPUT
------
- dataset_fair_eval/test/{images,labels}  (citra hard-link, label dicopy)
- dataset_fair_eval/data.yaml             (train/val/test SEMUA menunjuk test/images,
                                           supaya --split apa pun bisa dipakai evaluate.py)
- research/results/fair_eval_manifest.csv (daftar frame + alasan disertakan/dibuang)

CARA MENJALANKAN
----------------
    uv run python research/make_fair_eval.py --verify   # bandingkan dengan yang sudah ada
    uv run python research/make_fair_eval.py            # bangun (menolak menimpa)
    uv run python research/make_fair_eval.py --force     # bangun ulang, hapus yang lama

Setelah dibangun, jalankan S3-redo agar artefak evaluasinya ada:
    uv run python scripts/evaluate.py -w models/best_v2_backup.pt \\
        -d dataset_fair_eval/data.yaml -s test -n fair_V2_live
    uv run python scripts/evaluate.py -w runs/yolo26s_det_v21_clean/weights/best.pt \\
        -d dataset_fair_eval/data.yaml -s test -n fair_V21_clean
(catatan: `models/best_v2_backup.pt` ada di repo backend, sesuaikan path bila perlu)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "research" / "results"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES = ["habis", "hampir habis", "meja", "penuh", "sedikit"]

# Nomor screenshot = identitas frame sumber, stabil lintas ekspor Roboflow.
FRAME_RE = re.compile(r"Screenshot \((\d+)\)")


def frame_id(name: str) -> int | None:
    """Ambil nomor screenshot dari nama berkas. None bila polanya tidak cocok."""
    m = FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def collect(split_dir: Path) -> dict[int, Path]:
    """Peta {nomor_frame: path citra} untuk satu direktori images."""
    out: dict[int, Path] = {}
    if not split_dir.is_dir():
        return out
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        fid = frame_id(p.name)
        if fid is None:
            print(f"[warn] tidak bisa membaca nomor frame dari {p.name} - dilewati")
            continue
        if fid in out:
            print(f"[warn] nomor frame {fid} muncul dua kali di {split_dir} - memakai yang pertama")
            continue
        out[fid] = p
    return out


def v2_all_frames(v2_root: Path) -> tuple[set[int], dict[str, set[int]]]:
    """Semua nomor frame yang muncul di ekspor V2, beserta rinciannya per split."""
    per_split = {s: set(collect(v2_root / s / "images")) for s in SPLITS}
    return set().union(*per_split.values()), per_split


def label_for(img: Path) -> Path | None:
    """Berkas label pendamping sebuah citra (../labels/<stem>.txt)."""
    cand = img.parent.parent / "labels" / f"{img.stem}.txt"
    return cand if cand.is_file() else None


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link supaya tidak menggandakan disk; copy bila lintas volume."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun ulang himpunan uji adil (S3).")
    ap.add_argument("--clean", default="dataset_det_clean", help="dataset deteksi split bersih")
    ap.add_argument("--v2", default="v2-gorengan-counter.yolo26", help="ekspor V2 (untuk daftar pengecualian)")
    ap.add_argument("--dst", default="dataset_fair_eval", help="direktori keluaran")
    ap.add_argument("--verify", action="store_true", help="hanya bandingkan dengan yang sudah ada")
    ap.add_argument("--force", action="store_true", help="hapus direktori keluaran lama")
    args = ap.parse_args()

    clean_root = (REPO_ROOT / args.clean).resolve()
    v2_root = (REPO_ROOT / args.v2).resolve()
    dst_root = (REPO_ROOT / args.dst).resolve()

    if not (clean_root / "test" / "images").is_dir():
        sys.exit(f"ERROR: tidak ada {clean_root / 'test' / 'images'}")
    if not v2_root.is_dir():
        sys.exit(
            f"ERROR: ekspor V2 tidak ditemukan: {v2_root}\n"
            "Daftar pengecualian HARUS diturunkan dari ekspor V2 - jangan ditulis manual."
        )

    clean_test = collect(clean_root / "test" / "images")
    v2_frames, v2_per_split = v2_all_frames(v2_root)

    excluded = sorted(set(clean_test) & v2_frames)
    kept = sorted(set(clean_test) - v2_frames)

    print(f"clean test          : {len(clean_test)} frame")
    print(f"ekspor V2 (semua)   : {len(v2_frames)} frame"
          f"  (train {len(v2_per_split['train'])}, valid {len(v2_per_split['valid'])},"
          f" test {len(v2_per_split['test'])})")
    print(f"dibuang (irisan)    : {len(excluded)} -> {excluded}")
    print(f"dipakai untuk uji   : {len(kept)} frame\n")

    # Rincian alasan tiap pengecualian - inilah yang membuat aturannya dapat diaudit.
    print("Rincian alasan pengecualian:")
    for fid in excluded:
        where = [s for s in SPLITS if fid in v2_per_split[s]]
        print(f"  {fid}: ada di V2 {'+'.join(where)}")
    train_valid_only = sorted(set(clean_test) & (v2_per_split["train"] | v2_per_split["valid"]))
    print(f"\nCatatan dokumentasi: aturan 'trained or validated' saja menghasilkan"
          f" {len(train_valid_only)} frame ({train_valid_only}), BUKAN {len(excluded)}."
          f" Paper harus memakai kalimat 'setiap frame yang muncul pada ekspor V2'.")

    if args.verify:
        existing = collect(dst_root / "test" / "images")
        if not existing:
            print(f"\n[verify] {dst_root} kosong/tidak ada - tidak ada yang dibandingkan.")
            return
        same = sorted(existing) == kept
        print(f"\n[verify] yang ada di disk: {len(existing)} frame")
        print(f"[verify] hasil hitungan  : {len(kept)} frame")
        if same:
            print("[verify] IDENTIK - artefak di disk dapat direproduksi oleh script ini.")
        else:
            only_disk = sorted(set(existing) - set(kept))
            only_calc = sorted(set(kept) - set(existing))
            print("[verify] BERBEDA!")
            if only_disk:
                print(f"  hanya di disk     : {only_disk}")
            if only_calc:
                print(f"  hanya di hitungan : {only_calc}")
            sys.exit(1)
        return

    if dst_root.exists():
        if not args.force:
            sys.exit(
                f"ERROR: {dst_root} sudah ada. Jalankan dengan --verify untuk membandingkan,"
                " atau --force untuk membangun ulang."
            )
        print(f"[force] menghapus {dst_root}")
        shutil.rmtree(dst_root)

    missing_labels = 0
    for fid in kept:
        img = clean_test[fid]
        link_or_copy(img, dst_root / "test" / "images" / img.name)
        lbl = label_for(img)
        if lbl is None:
            missing_labels += 1
            print(f"[warn] frame {fid} ({img.name}) tidak punya label")
            continue
        dst_lbl = dst_root / "test" / "labels" / lbl.name
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lbl, dst_lbl)

    # train/val/test semuanya menunjuk test/images: disengaja, supaya evaluate.py bisa
    # dipanggil dengan --split apa pun tanpa menyiapkan split kosong.
    names = ", ".join(f"'{c}'" for c in CLASS_NAMES)
    (dst_root / "data.yaml").write_text(
        "# fair eval: clean test split minus every frame present in the V2 export"
        f" ({len(excluded)} removed: {', '.join(str(f) for f in excluded)})\n"
        f"path: {dst_root.as_posix()}\n"
        "train: test/images\n"
        "val: test/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_DIR / "fair_eval_manifest.csv"
    try:
        with open(manifest, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "status", "alasan", "berkas"])
            for fid in sorted(clean_test):
                if fid in excluded:
                    where = "+".join(s for s in SPLITS if fid in v2_per_split[s])
                    w.writerow([fid, "dibuang", f"ada di ekspor V2 ({where})", clean_test[fid].name])
                else:
                    w.writerow([fid, "dipakai", "tidak ada di ekspor V2", clean_test[fid].name])
    except OSError as exc:
        print(f"PERINGATAN: gagal menulis manifest: {exc}")

    n_img = len(list((dst_root / "test" / "images").iterdir()))
    n_lbl = len(list((dst_root / "test" / "labels").iterdir())) if (dst_root / "test" / "labels").is_dir() else 0
    print(f"\nSelesai: {dst_root}")
    print(f"  citra {n_img}, label {n_lbl}" + (f", label hilang {missing_labels}" if missing_labels else ""))
    print(f"  manifest: {manifest}")
    print("\nLangkah berikutnya (S3-redo, evaluasi saja - bukan pelatihan):")
    print("  uv run python scripts/evaluate.py -w <bobot V2> -d dataset_fair_eval/data.yaml"
          " -s test -n fair_V2_live")
    print("  uv run python scripts/evaluate.py -w runs/yolo26s_det_v21_clean/weights/best.pt"
          " -d dataset_fair_eval/data.yaml -s test -n fair_V21_clean")


if __name__ == "__main__":
    main()
