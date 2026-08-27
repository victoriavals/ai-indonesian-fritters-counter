"""Pantau progres pelatihan - satu perintah, keluaran bersih.

KENAPA SCRIPT INI ADA
---------------------
Berkas log mentah (`research/logs/*.log`) memuat kode warna ANSI dan gambar-ulang progress bar
Ultralytics, sehingga `tail -f` menampilkan kekacauan yang sulit dibaca. Sumber progres yang
paling bersih adalah `runs/<nama>/results.csv`, yang ditulis Ultralytics satu baris per epoch
langsung ke disk - tidak terpengaruh buffering pipa.

CARA MENJALANKAN
----------------
    uv run python research/status.py            # ringkasan semua run
    uv run python research/status.py --watch     # perbarui tiap 30 detik
    uv run python research/status.py --tail 8    # plus 8 epoch terakhir run aktif
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"

# Urutan sesuai PLAN di run_experiments.py
URUTAN = [
    "yolo11s_det_v21_clean",
    "yolov8s_det_v21_clean",
    "yolo26n_det_v21_clean",
    "yolo26s_det_v21_clean_seed1",
    "yolo26s_det_v21_clean_seed2",
]
TOTAL_EPOCH = 200


def baca(nama: str) -> dict | None:
    d = RUNS / nama
    csv_path = d / "results.csv"
    if not csv_path.is_file():
        return None
    try:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    def f(row, kunci):
        try:
            return float(row[kunci])
        except (KeyError, ValueError, TypeError):
            return 0.0

    # fitness Ultralytics = 0,1*mAP50 + 0,9*mAP50-95 - kriteria pemilihan model terbaik
    fit = [(0.1 * f(r, "metrics/mAP50(B)") + 0.9 * f(r, "metrics/mAP50-95(B)"),
            int(f(r, "epoch"))) for r in rows]
    terbaik = max(fit)
    detik = f(rows[-1], "time")
    n = len(rows)

    # `run_meta.json` ditulis train.py hanya SETELAH pelatihan tuntas - bukan best.pt,
    # yang sudah ada sejak epoch pertama.
    tuntas = (d / "run_meta.json").is_file()

    return {
        "epoch": n,
        "mAP50": f(rows[-1], "metrics/mAP50(B)"),
        "mAP50_95": f(rows[-1], "metrics/mAP50-95(B)"),
        "fit_terbaik": terbaik[0],
        "epoch_terbaik": terbaik[1],
        "jam": detik / 3600,
        "menit_per_epoch": (detik / 60 / n) if n else 0.0,
        "tuntas": tuntas,
        "diubah": datetime.fromtimestamp(csv_path.stat().st_mtime),
    }


def tampil(tail: int) -> None:
    print(f"\n  {datetime.now():%H:%M:%S}  progres pelatihan\n")
    print(f"  {'run':32s}{'epoch':>9s}{'mAP50':>8s}{'terbaik':>9s}{'jam':>7s}  status")
    print("  " + "-" * 76)

    aktif = None
    selesai = 0
    for nama in URUTAN:
        d = baca(nama)
        if d is None:
            print(f"  {nama:32s}{'-':>9s}{'-':>8s}{'-':>9s}{'-':>7s}  belum mulai")
            continue
        if d["tuntas"]:
            status = "TUNTAS"
            selesai += 1
        else:
            # Bila results.csv tidak berubah > 5 menit, kemungkinan proses berhenti.
            diam = (datetime.now() - d["diubah"]).total_seconds()
            status = "berjalan" if diam < 300 else f"DIAM {diam / 60:.0f} menit"
            aktif = (nama, d)
        print(f"  {nama:32s}{d['epoch']:>5d}/{TOTAL_EPOCH}{d['mAP50']:>8.4f}"
              f"{d['epoch_terbaik']:>9d}{d['jam']:>7.2f}  {status}")

    print("  " + "-" * 76)
    print(f"  tuntas: {selesai}/{len(URUTAN)}")

    if aktif:
        nama, d = aktif
        sisa_epoch = TOTAL_EPOCH - d["epoch"]
        sisa_run = len(URUTAN) - selesai - 1
        # Perkiraan kasar: early stopping biasanya memotong di ~135 epoch pada dataset ini,
        # jadi ETA memakai 135 sebagai target, bukan 200.
        target = 135
        sisa_aktif = max(0, target - d["epoch"]) * d["menit_per_epoch"] / 60
        print(f"\n  aktif : {nama} ({d['menit_per_epoch']:.2f} menit/epoch)")
        print(f"  ETA   : ~{sisa_aktif:.1f} jam untuk run ini"
              f" + ~{sisa_run * target * d['menit_per_epoch'] / 60:.1f} jam untuk {sisa_run} run sisa")
        print(f"          (target {target} epoch; early stopping dapat memotong lebih awal,"
              " batas keras 200)")

        if tail:
            csv_path = RUNS / nama / "results.csv"
            with open(csv_path, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            print(f"\n  {tail} epoch terakhir {nama}:")
            print(f"    {'epoch':>6s}{'mAP50':>9s}{'mAP50-95':>10s}{'box_loss':>10s}")
            for r in rows[-tail:]:
                print(f"    {r['epoch']:>6s}{float(r['metrics/mAP50(B)']):>9.4f}"
                      f"{float(r['metrics/mAP50-95(B)']):>10.4f}"
                      f"{float(r['train/box_loss']):>10.4f}")
    elif selesai == len(URUTAN):
        print("\n  SELURUH RANGKAIAN TUNTAS - lanjutkan ke evaluasi dan benchmark.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pantau progres pelatihan.")
    ap.add_argument("--watch", action="store_true", help="perbarui tiap 30 detik")
    ap.add_argument("--tail", type=int, default=0, help="tampilkan N epoch terakhir run aktif")
    args = ap.parse_args()

    if not args.watch:
        tampil(args.tail)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")   # bersihkan layar
            tampil(args.tail)
            time.sleep(30)
    except KeyboardInterrupt:
        print("  dihentikan.\n")


if __name__ == "__main__":
    main()
