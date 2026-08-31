"""Bangun tabel hyperparameter naskah LANGSUNG dari `runs/<nama>/args.yaml`.

Tidak ada satu pun nilai yang diketik ulang di sini -- semuanya dibaca dari berkas yang
Ultralytics tulis saat pelatihan berjalan. Itu aturan yang sama dengan `generate_tables.py`:
angka di naskah harus dapat dilacak ke artefak, bukan ke ingatan.

DUA HAL YANG TIDAK BOLEH DILAPORKAN MENTAH DARI args.yaml
---------------------------------------------------------
1. `optimizer: auto` membuat Ultralytics MENGABAIKAN `lr0` dan `momentum` di berkas itu, lalu
   menentukan sendiri optimizer dan learning rate-nya. Melaporkan lr0=0,01 dan momentum=0,937
   apa adanya akan salah. Nilai efektifnya dibaca dari log pelatihan (baris "optimizer: AdamW(...)")
   dan diverifikasi konsisten di seluruh run.
2. `box`/`cls`/`dfl` adalah bobot loss bawaan Ultralytics. Dicatat apa adanya tanpa klaim
   mengenai komponen mana yang aktif pada arsitektur tertentu.

Pakai:
    uv run python research/generate_table_hyperparam.py
    uv run python research/generate_table_hyperparam.py --verify   # cek keseragaman antar-run
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
LOGS = REPO_ROOT / "research" / "logs"
OUT = REPO_ROOT.parent / "Jurnal" / "tables"

RUN_UTAMA = "yolo26s_det_v21_clean"
RUN_LAIN = [
    "yolo11s_det_v21_clean",
    "yolov8s_det_v21_clean",
    "yolo26n_det_v21_clean",
    "yolo26s_det_v21_clean_seed1",
    "yolo26s_det_v21_clean_seed2",
]

# Kunci yang memang BOLEH berbeda antar-run: itulah variabel percobaannya.
# CATATAN: `data` sengaja TIDAK ada di sini. Bila dua run memakai dataset berbeda,
# perbandingannya batal - itu harus terdeteksi, bukan dimaafkan.
BOLEH_BEDA = {"model", "name", "seed", "save_dir", "project", "time"}


def koma(v) -> str:
    """Format angka dengan koma desimal (kaidah penulisan Bahasa Indonesia)."""
    if isinstance(v, bool):
        return "ya" if v else "tidak"
    if isinstance(v, float):
        s = f"{v:g}"
        return s.replace(".", ",")
    return str(v)


def optimizer_efektif() -> tuple[str, str, str]:
    """Baca optimizer yang BENAR-BENAR dipakai dari log pelatihan."""
    pola = re.compile(r"optimizer:.*?(\w+)\(lr=([\d.]+), momentum=([\d.]+)\)")
    temuan = set()
    for log in sorted(LOGS.glob("*.log")):
        for baris in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pola.search(baris)
            if m:
                temuan.add((m.group(1), m.group(2), m.group(3)))
    if not temuan:
        raise SystemExit(
            "GAGAL: tidak menemukan baris 'optimizer: <nama>(lr=..., momentum=...)' di "
            f"{LOGS}. Tanpa itu nilai efektif tidak dapat diverifikasi, dan args.yaml "
            "TIDAK boleh dilaporkan mentah karena optimizer=auto mengabaikannya."
        )
    if len(temuan) > 1:
        raise SystemExit(f"GAGAL: optimizer efektif tidak seragam antar-run: {sorted(temuan)}")
    return temuan.pop()


def muat(run: str) -> dict:
    p = RUNS / run / "args.yaml"
    if not p.is_file():
        raise SystemExit(f"GAGAL: {p} tidak ada.")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def verifikasi(a: dict) -> list[str]:
    """Pastikan seluruh run identik kecuali pada kunci yang memang divariasikan."""
    beda: list[str] = []
    for run in RUN_LAIN:
        b = muat(run)
        for k in sorted(set(a) | set(b)):
            if k in BOLEH_BEDA:
                continue
            if a.get(k) != b.get(k):
                beda.append(f"{run}: {k} = {a.get(k)!r} vs {b.get(k)!r}")
    return beda


def baris_tabel(a: dict) -> list[tuple[str, str, str]]:
    opt, lr, mom = optimizer_efektif()
    K = "Konfigurasi umum"
    O = "Optimisasi"
    G = "Augmentasi data"
    return [
        (K, "Model dasar", f"{a['model']} (pralatih)"),
        (K, "Ukuran masukan (piksel)", koma(a["imgsz"])),
        (K, "Epoch maksimum", koma(a["epochs"])),
        (K, "Patience (henti dini)", koma(a["patience"])),
        (K, "Ukuran batch", koma(a["batch"])),
        (K, "Nominal batch size", koma(a["nbs"])),
        (K, "Presisi campuran (AMP)", koma(a["amp"])),
        (K, "Mode deterministik", koma(a["deterministic"])),
        (K, "Seed", koma(a["seed"])),

        (O, "Optimizer", f"{opt} (dipilih otomatis)"),
        (O, "Learning rate awal", koma(float(lr))),
        (O, "Faktor learning rate akhir", koma(a["lrf"])),
        (O, "Momentum", koma(float(mom))),
        (O, "Weight decay", koma(a["weight_decay"])),
        (O, "Penjadwal cosine", koma(a["cos_lr"])),
        (O, "Warmup (epoch)", koma(a["warmup_epochs"])),
        (O, "Warmup momentum", koma(a["warmup_momentum"])),
        (O, "Warmup bias lr", koma(a["warmup_bias_lr"])),
        (O, "Bobot loss box / cls / dfl",
            f"{koma(a['box'])} / {koma(a['cls'])} / {koma(a['dfl'])}"),

        (G, "HSV hue / saturation / value",
            f"{koma(a['hsv_h'])} / {koma(a['hsv_s'])} / {koma(a['hsv_v'])}"),
        (G, "Translasi", koma(a["translate"])),
        (G, "Skala", koma(a["scale"])),
        (G, "Pencerminan horizontal", koma(a["fliplr"])),
        (G, "Pencerminan vertikal", koma(a["flipud"])),
        (G, "Rotasi / shear / perspektif",
            f"{koma(a['degrees'])} / {koma(a['shear'])} / {koma(a['perspective'])}"),
        (G, "Mosaic", f"{koma(a['mosaic'])} (dimatikan {koma(a['close_mosaic'])} epoch terakhir)"),
        (G, "Mixup / cutmix / copy-paste",
            f"{koma(a['mixup'])} / {koma(a['cutmix'])} / {koma(a['copy_paste'])}"),
        (G, "Random erasing", koma(a["erasing"])),
        (G, "Auto augment", str(a["auto_augment"])),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="hanya periksa keseragaman antar-run, tanpa menulis berkas")
    args = ap.parse_args()

    a = muat(RUN_UTAMA)
    beda = verifikasi(a)

    if args.verify:
        if beda:
            print("BERBEDA di luar kunci yang dimaksud:")
            for b in beda:
                print("  -", b)
            raise SystemExit(1)
        print(f"[OK] {len(RUN_LAIN) + 1} run identik kecuali pada: {sorted(BOLEH_BEDA)}")
        return

    if beda:
        raise SystemExit(
            "GAGAL: run tidak seragam, keterangan tabel tidak akan jujur. Jalankan "
            "--verify untuk melihat rinciannya."
        )

    rows = baris_tabel(a)
    OUT.mkdir(parents=True, exist_ok=True)

    csv_p = OUT / "tabel-hyperparameter.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Kelompok", "Parameter", "Nilai"])
        w.writerows(rows)

    md = ["**Tabel X. Hyperparameter Pelatihan**", "",
          "| Kelompok | Parameter | Nilai |", "|---|---|---|"]
    sebelumnya = None
    for kel, par, nil in rows:
        md.append(f"| {kel if kel != sebelumnya else ''} | {par} | {nil} |")
        sebelumnya = kel
    md += ["", "Sumber: Data primer yang diolah, 2026", "",
           "_Seluruh nilai dibaca dari `args.yaml` yang ditulis Ultralytics saat pelatihan. "
           f"Konfigurasi identik pada {len(RUN_LAIN) + 1} run; yang divariasikan hanya model "
           "dasar (perbandingan arsitektur) dan seed (pengukuran ragam antar-run). "
           "Karena `optimizer=auto`, nilai learning rate dan momentum yang dilaporkan adalah "
           "nilai efektif dari log pelatihan, bukan nilai `lr0`/`momentum` di `args.yaml` yang "
           "diabaikan Ultralytics._"]
    (OUT / "tabel-hyperparameter.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] {len(rows)} baris -> {csv_p.name} + tabel-hyperparameter.md")
    print(f"[OK] keseragaman {len(RUN_LAIN) + 1} run terverifikasi")


if __name__ == "__main__":
    main()
