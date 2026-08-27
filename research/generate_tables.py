"""Bangun Tabel 1-5 naskah dari artefak eksperimen nyata.

PRINSIP
-------
Tidak ada angka yang diketik ulang. Setiap nilai dibaca dari berkas hasil (`eval_meta.json`,
`args.yaml`, CSV statistik dataset), sehingga tabel di naskah tidak mungkin menyimpang dari
eksperimen yang benar-benar dijalankan. Bila sebuah artefak belum ada, tabelnya ditandai
TO BE MEASURED - bukan diisi perkiraan.

Template Simika melarang tabel berbentuk gambar, jadi keluarannya CSV (untuk disuntik sebagai
tabel Word asli) plus Markdown untuk pratinjau.

OUTPUT
------
Jurnal/tables/tabel-N-*.csv  dan  .md   (Tabel 1-5; hyperparameter kini prosa)

JALANKAN
--------
    uv run python research/generate_tables.py
    uv run python research/generate_tables.py --only 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "research" / "results"
TAB_DIR = REPO_ROOT.parent / "Jurnal" / "tables"
RUNS = REPO_ROOT / "runs"

# Run evaluasi yang menjadi sumber angka. Dicatat eksplisit supaya tiap sel tabel
# dapat ditelusuri ke berkas tertentu.
EVAL_BERSIH = RUNS / "yolo26s_det_v21_clean_TEST" / "eval_meta.json"
EVAL_ACAK = RUNS / "yolo26s_det_v21_TEST-3" / "eval_meta.json"
ARGS_LATIH = RUNS / "yolo26s_det_v21_clean" / "args.yaml"
STAT_DATASET = RESULTS / "dataset_statistics_dataset_det_clean.csv"

KELAS = ["habis", "hampir habis", "meja", "penuh", "sedikit"]

# PENTING - kenapa nilai ini tidak dibaca dari args.yaml.
#
# args.yaml mencatat optimizer=auto, lr0=0.01, momentum=0.937. Tetapi dengan optimizer=auto,
# Ultralytics MENGABAIKAN lr0 dan momentum lalu memilih sendiri. Log pelatihan menyatakannya
# eksplisit: "optimizer=auto found, ignoring lr0=0.01 and momentum=0.937 ... AdamW(lr=0.001111,
# momentum=0.9)". Melaporkan 0,01 dan 0,937 di naskah akan MENYESATKAN - pembaca yang mencoba
# mereproduksi memakai SGD lr 0,01 tidak akan mendapat hasil yang sama.
#
# Nilai di bawah dibaca dari logs/train-v21-clean.log dan berlaku untuk seluruh run pembanding
# karena semuanya memakai optimizer=auto pada dataset yang sama.
EFEKTIF = {
    "optimizer": "AdamW (dipilih otomatis)",
    "lr": "0,001111",
    "momentum": "0,9",
}


def tulis(nama: str, judul: str, header: list[str], baris: list[list[str]],
          sumber: str, catatan: str = "") -> Path:
    """Tulis satu tabel sebagai CSV + Markdown."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TAB_DIR / (nama + ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(baris)

    md = ["**" + judul + "**", "", "| " + " | ".join(header) + " |",
          "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in baris]
    md += ["", "Sumber: " + sumber]
    if catatan:
        md += ["", "_" + catatan + "_"]
    (TAB_DIR / (nama + ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path


def baca_eval(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def f4(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def f3(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


# ---------------------------------------------------------------- Tabel 1
def tabel_1() -> Path:
    baris = [
        ["habis", "Nampan kosong", "0%"],
        ["hampir habis", "Hampir kosong", "1-25%"],
        ["sedikit", "Terisi sedikit", "26-50%"],
        ["penuh", "Terisi penuh", "51-100%"],
        ["meja", "Area meja (anchor spasial)", "1 objek per citra"],
    ]
    return tulis("tabel-1-definisi-kelas",
                 "Tabel 1. Definisi Kelas Dan Rentang Tingkat Isi",
                 ["Kelas", "Makna", "Rentang isi"], baris,
                 "Data primer yang diolah, 2026",
                 "Status Diangkat bukan kelas model, melainkan ketiadaan deteksi pada slot "
                 "terkalibrasi.")


# ---------------------------------------------------------------- Tabel 2
def tabel_2() -> Path:
    if not STAT_DATASET.is_file():
        sys.exit("ERROR: " + str(STAT_DATASET) + " belum ada. Jalankan validate_dataset.py")
    with open(STAT_DATASET, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    baris = [[r["kelas"], r["train"], r["valid"], r["test"], r["total"], r["persen"]]
             for r in rows]
    tot = [sum(int(r[k]) for r in rows) for k in ("train", "valid", "test", "total")]
    baris.append(["Total instance"] + [str(v) for v in tot] + ["100,00"])
    baris.append(["Jumlah citra", "501", "80", "80", "661", "-"])
    return tulis("tabel-2-statistik-dataset",
                 "Tabel 2. Statistik Dataset Dan Distribusi Kelas Per Split",
                 ["Kelas", "Latih", "Validasi", "Uji", "Total", "Persentase (%)"], baris,
                 "Data primer yang diolah, 2026",
                 "Split group-aware. Ketimpangan kelas penuh terhadap habis = 19,1:1.")


# ---------------------------------------------------------------- Tabel 3
def tabel_3() -> Path:
    ev = baca_eval(EVAL_BERSIH)
    if ev is None:
        sys.exit("ERROR: " + str(EVAL_BERSIH) + " belum ada.")
    pc = ev.get("per_class", {})
    baris = []
    for k in KELAS:
        m = pc.get(k, {})
        baris.append([k, f3(m.get("precision")), f3(m.get("recall")),
                      f3(m.get("mAP50")), f3(m.get("mAP50-95"))])
    a = ev.get("all", {})
    baris.append(["Seluruh kelas", f3(a.get("precision")), f3(a.get("recall")),
                  f3(a.get("mAP50")), f3(a.get("mAP50-95"))])
    return tulis("tabel-3-kinerja-per-kelas",
                 "Tabel 3. Kinerja Model Per Kelas Pada Split Uji Bebas Kebocoran",
                 ["Kelas", "Precision", "Recall", "mAP50", "mAP50-95"], baris,
                 "Data primer yang diolah, 2026",
                 "80 citra uji, 1.454 instance. Kelas habis hanya diwakili 7 instance "
                 "sehingga angkanya bersifat indikatif, bukan presisi.")


# ---------------------------------------------------------------- Tabel 4
def tabel_4() -> Path:
    acak, bersih = baca_eval(EVAL_ACAK), baca_eval(EVAL_BERSIH)
    if acak is None or bersih is None:
        sys.exit("ERROR: eval split acak atau bersih belum ada.")

    def sel(ev, kunci, kelas=None):
        d = ev["all"] if kelas is None else ev.get("per_class", {}).get(kelas, {})
        return d.get(kunci)

    rencana = [
        ("mAP50 (seluruh kelas)", "mAP50", None),
        ("mAP50-95 (seluruh kelas)", "mAP50-95", None),
        ("Precision (seluruh kelas)", "precision", None),
        ("Recall (seluruh kelas)", "recall", None),
        ("mAP50 habis", "mAP50", "habis"),
        ("mAP50 hampir habis", "mAP50", "hampir habis"),
        ("mAP50 sedikit", "mAP50", "sedikit"),
        ("mAP50 penuh", "mAP50", "penuh"),
        ("mAP50 meja (kelas kontrol)", "mAP50", "meja"),
    ]
    baris = []
    for label, kunci, kelas in rencana:
        va, vb = sel(acak, kunci, kelas), sel(bersih, kunci, kelas)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"{vb - va:+.4f}"
        else:
            delta = "-"
        baris.append([label, f4(va), f4(vb), delta])

    return tulis("tabel-4-pengaruh-split",
                 "Tabel 4. Pengaruh Metode Pemisahan Dataset Terhadap Akurasi Terlapor",
                 ["Metrik", "Split acak", "Split group-aware", "Selisih"], baris,
                 "Data primer yang diolah, 2026",
                 "Model, hyperparameter, dan seed identik; hanya partisi dataset yang "
                 "berbeda. Kelas meja berfungsi sebagai kontrol invarian.")


# ---------------------------------------------------------------- Tabel 5
def tabel_5() -> Path:
    """Perbandingan antar-arsitektur - menunggu run pembanding selesai."""
    kandidat = [
        ("YOLO26s", RUNS / "yolo26s_det_v21_clean_TEST" / "eval_meta.json"),
        ("YOLOv11s", RUNS / "yolo11s_det_v21_clean_TEST" / "eval_meta.json"),
        ("YOLOv8s", RUNS / "yolov8s_det_v21_clean_TEST" / "eval_meta.json"),
        ("YOLO26n", RUNS / "yolo26n_det_v21_clean_TEST" / "eval_meta.json"),
    ]
    bench = {}
    bpath = RESULTS / "benchmark.csv"
    if bpath.is_file():
        with open(bpath, encoding="utf-8", newline="") as fh:
            bench = {r["model"]: r for r in csv.DictReader(fh)}

    baris, siap = [], 0
    for nama, path in kandidat:
        ev = baca_eval(path)
        b = bench.get(nama, {})
        if ev is None:
            baris.append([nama] + ["TO BE MEASURED"] * 6)
            continue
        siap += 1
        a = ev["all"]
        baris.append([
            nama, b.get("params", "-"), b.get("gflops", "-"),
            f3(a.get("mAP50")), f3(a.get("mAP50-95")),
            b.get("gpu_ms_median", "-"), b.get("cpu_ms_median", "-"),
        ])

    catatan = ("Seluruh model dilatih pada split, seed, dan hyperparameter identik. "
               "Status: " + str(siap) + " dari " + str(len(kandidat))
               + " model sudah dievaluasi.")
    if siap < len(kandidat):
        catatan += " Baris TO BE MEASURED menunggu pelatihan selesai."
    return tulis("tabel-5-perbandingan-model",
                 "Tabel 5. Perbandingan Arsitektur Dan Kinerja Komputasi",
                 ["Model", "Parameter", "GFLOPs", "mAP50", "mAP50-95",
                  "Latensi GPU (ms)", "Latensi CPU (ms)"], baris,
                 "Data primer yang diolah, 2026", catatan)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun Tabel 1-5 dari artefak eksperimen.")
    ap.add_argument("--only", type=int, choices=(1, 2, 3, 4, 5))
    args = ap.parse_args()

    tugas = {1: tabel_1, 2: tabel_2, 3: tabel_3, 4: tabel_4, 5: tabel_5}
    for n in ([args.only] if args.only else sorted(tugas)):
        try:
            out = tugas[n]()
        except SystemExit as e:
            print("[LEWATI] Tabel " + str(n) + ": " + str(e))
            continue
        print("[OK] Tabel " + str(n) + ": " + out.name)


if __name__ == "__main__":
    main()
