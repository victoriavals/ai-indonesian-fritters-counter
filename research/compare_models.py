"""Bangun tabel perbandingan model (T7 dan T8) dari artefak `eval_meta.json`.

KENAPA SCRIPT INI ADA
---------------------
Blok "PERBANDINGAN ADIL" di `logs/fair-eval.log` dibuat oleh script ad-hoc yang tidak pernah
masuk repo. Artinya tabel T7 tidak dapat dibangun ulang. Script ini menggantikannya, dan
sekaligus dipakai untuk T8 (perbandingan antar-arsitektur) begitu run pembanding selesai.

Semua angka dibaca dari `eval_meta.json` yang ditulis `scripts/evaluate.py` - tidak ada angka
yang diketik ulang, sehingga tabel tidak bisa menyimpang dari hasil eksperimen.

INPUT
-----
- runs/<nama>/eval_meta.json   (dihasilkan scripts/evaluate.py)
- opsional: research/results/benchmark.csv (dari research/benchmark.py) untuk kolom efisiensi
- opsional: runs/<nama>/results.csv dan run_meta.json untuk jumlah epoch

OUTPUT
------
- research/results/<out>.csv        tabel utama (keseluruhan + per kelas)
- research/results/<out>.tex        versi LaTeX siap tempel
- research/results/<out>.md         versi Markdown untuk pratinjau
- ringkasan ke stdout

CARA MENJALANKAN
----------------
    # T7 - uji adil dua model, dengan kolom selisih
    uv run python research/compare_models.py \\
        --runs fair_V2_live fair_V21_clean \\
        --labels "V2 (lama)" "V2.1-clean" \\
        --delta --out T7_uji_adil

    # T8 - perbandingan antar-arsitektur (setelah run pembanding selesai + dievaluasi)
    uv run python research/compare_models.py \\
        --runs yolo26s_det_v21_clean_TEST yolo11s_det_v21_clean_TEST yolov8s_det_v21_clean_TEST \\
        --labels YOLO26s YOLOv11s YOLOv8s \\
        --benchmark --out T8_perbandingan_model

CATATAN
-------
`--delta` hanya sah untuk TEPAT DUA run, dan selisih dihitung sebagai (run kedua - run pertama).
Untuk T8 jangan pakai --delta: membandingkan banyak model butuh tabel, bukan satu selisih.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "research" / "results"

METRICS = ("precision", "recall", "mAP50", "mAP50-95")
METRIC_LABEL = {"precision": "P", "recall": "R", "mAP50": "mAP50", "mAP50-95": "mAP50-95"}


def load_eval(run: str) -> dict:
    """Baca eval_meta.json satu run. Berhenti dengan pesan jelas bila tidak ada."""
    path = RUNS_DIR / run / "eval_meta.json"
    if not path.is_file():
        sys.exit(
            f"ERROR: {path} tidak ada.\n"
            f"Jalankan dulu evaluasinya, misalnya:\n"
            f"  uv run python scripts/evaluate.py -w runs/<run-latih>/weights/best.pt \\\n"
            f"      -d dataset_det_clean/data.yaml -s test -n {run}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: gagal membaca {path}: {exc}")


def epochs_of(run_name: str) -> str:
    """Jumlah epoch dari results.csv run PELATIHAN (bukan run evaluasi)."""
    # Run evaluasi biasanya bernama <run-latih>_TEST atau nama bebas; coba tebak.
    for cand in (run_name, run_name.replace("_TEST", "")):
        csv_path = RUNS_DIR / cand / "results.csv"
        if csv_path.is_file():
            try:
                with open(csv_path, encoding="utf-8") as fh:
                    return str(max(0, sum(1 for _ in fh) - 1))
            except OSError:
                return "-"
    return "-"


def load_benchmark() -> dict[str, dict[str, str]]:
    """Peta {label_model: baris benchmark} bila research/results/benchmark.csv ada."""
    path = RESULTS_DIR / "benchmark.csv"
    if not path.is_file():
        print(f"[info] {path} tidak ada - kolom efisiensi dilewati"
              " (jalankan research/benchmark.py untuk mengisinya)")
        return {}
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return {row["model"]: row for row in csv.DictReader(fh)}
    except (OSError, KeyError) as exc:
        print(f"[warn] gagal membaca {path}: {exc}")
        return {}


def fmt(v: object, nd: int = 4) -> str:
    """Format angka konsisten; non-angka dikembalikan apa adanya."""
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v) if v is not None else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun tabel perbandingan model dari eval_meta.json.")
    ap.add_argument("--runs", nargs="+", required=True, help="nama run di runs/ yang punya eval_meta.json")
    ap.add_argument("--labels", nargs="*", help="label tampilan per run (default: nama run)")
    ap.add_argument("--out", default="perbandingan_model", help="nama berkas keluaran tanpa ekstensi")
    ap.add_argument("--delta", action="store_true", help="tambah kolom selisih (hanya untuk 2 run)")
    ap.add_argument("--benchmark", action="store_true", help="gabungkan kolom efisiensi dari benchmark.csv")
    args = ap.parse_args()

    labels = args.labels or list(args.runs)
    if len(labels) != len(args.runs):
        sys.exit(f"ERROR: jumlah --labels ({len(labels)}) tidak sama dengan --runs ({len(args.runs)})")
    if args.delta and len(args.runs) != 2:
        sys.exit(f"ERROR: --delta hanya sah untuk tepat 2 run, diberikan {len(args.runs)}")

    metas = [load_eval(r) for r in args.runs]

    # Penjaga kesebandingan: semua run HARUS dievaluasi pada data + split yang sama,
    # kalau tidak tabelnya membandingkan hal yang berbeda tanpa terlihat.
    datasets = {Path(m.get("data", "?")).as_posix() for m in metas}
    splits = {m.get("split", "?") for m in metas}
    if len(datasets) > 1:
        print("PERINGATAN: run dievaluasi pada DATASET BERBEDA - tabel ini tidak sebanding:")
        for r, m in zip(args.runs, metas):
            print(f"    {r}: {m.get('data')}")
    if len(splits) > 1:
        print(f"PERINGATAN: split berbeda antar run: {splits}")

    bench = load_benchmark() if args.benchmark else {}

    # Kumpulkan nama kelas dari semua run (urutan stabil, mengikuti run pertama).
    class_names: list[str] = []
    for m in metas:
        for c in m.get("per_class", {}):
            if c not in class_names:
                class_names.append(c)

    # ---------- Susun baris tabel ----------
    rows: list[dict[str, str]] = []

    def add_row(scope: str, cls: str, getter) -> None:
        row: dict[str, str] = {"bagian": scope, "kelas": cls}
        vals: list[float | None] = []
        for label, m in zip(labels, metas):
            for met in METRICS:
                v = getter(m, met)
                row[f"{label} {METRIC_LABEL[met]}"] = fmt(v)
            vals.append(getter(m, "mAP50"))
        if args.delta and len(vals) == 2 and all(isinstance(v, (int, float)) for v in vals):
            row["selisih mAP50"] = f"{vals[1] - vals[0]:+.4f}"
        rows.append(row)

    add_row("keseluruhan", "all", lambda m, met: m.get("all", {}).get(met))
    for cls in class_names:
        add_row("per kelas", cls, lambda m, met, c=cls: m.get("per_class", {}).get(c, {}).get(met))

    # ---------- Cetak ke stdout ----------
    print(f"\nDataset : {datasets.pop() if len(datasets) == 1 else 'BERBEDA'}")
    print(f"Split   : {splits.pop() if len(splits) == 1 else 'BERBEDA'}\n")

    w_cls = max(14, max(len(c) for c in class_names + ['all']) + 2)
    head = f"{'kelas':{w_cls}s}" + "".join(f"{lb + ' mAP50':>22s}" for lb in labels)
    if args.delta:
        head += f"{'selisih':>12s}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['kelas']:{w_cls}s}"
        for lb in labels:
            line += f"{r.get(lb + ' mAP50', '-'):>22s}"
        if args.delta:
            line += f"{r.get('selisih mAP50', '-'):>12s}"
        print(line)

    if bench:
        print("\nEfisiensi:")
        print(f"  {'model':16s}{'params':>12s}{'GFLOPs':>10s}{'GPU ms':>10s}{'CPU ms':>10s}")
        for lb in labels:
            b = bench.get(lb, {})
            print(f"  {lb:16s}{b.get('params', '-'):>12s}{b.get('gflops', '-'):>10s}"
                  f"{b.get('gpu_ms_median', '-'):>10s}{b.get('cpu_ms_median', '-'):>10s}")

    print("\nEpoch pelatihan:")
    for r, lb in zip(args.runs, labels):
        print(f"  {lb:16s} {epochs_of(r):>5s} epoch   (run: {r})")

    # ---------- Tulis berkas ----------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bagian", "kelas"]
    for lb in labels:
        fieldnames += [f"{lb} {METRIC_LABEL[m]}" for m in METRICS]
    if args.delta:
        fieldnames.append("selisih mAP50")

    out_csv = RESULTS_DIR / f"{args.out}.csv"
    out_tex = RESULTS_DIR / f"{args.out}.tex"
    out_md = RESULTS_DIR / f"{args.out}.md"
    try:
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        # LaTeX - kolom mAP50 saja supaya muat di halaman jurnal; CSV memuat semuanya.
        cols = "l" + "r" * len(labels) + ("r" if args.delta else "")
        tex = ["\\begin{table}[htbp]", "\\centering",
               f"\\caption{{Perbandingan model ({args.out}).}}",
               f"\\label{{tab:{args.out}}}", f"\\begin{{tabular}}{{{cols}}}", "\\hline",
               "Kelas & " + " & ".join(labels) + (" & $\\Delta$" if args.delta else "") + " \\\\", "\\hline"]
        for r in rows:
            cells = [r["kelas"].replace("_", "\\_")] + [r.get(lb + " mAP50", "-") for lb in labels]
            if args.delta:
                cells.append(r.get("selisih mAP50", "-").replace("+", "$+$"))
            tex.append(" & ".join(cells) + " \\\\")
        tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
        out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")

        md = ["| Kelas | " + " | ".join(labels) + (" | Selisih |" if args.delta else " |"),
              "|---|" + "---|" * (len(labels) + (1 if args.delta else 0))]
        for r in rows:
            cells = [r["kelas"]] + [r.get(lb + " mAP50", "-") for lb in labels]
            if args.delta:
                cells.append(r.get("selisih mAP50", "-"))
            md.append("| " + " | ".join(cells) + " |")
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    except OSError as exc:
        sys.exit(f"ERROR: gagal menulis keluaran: {exc}")

    print(f"\nCSV   : {out_csv}")
    print(f"LaTeX : {out_tex}")
    print(f"MD    : {out_md}")


if __name__ == "__main__":
    main()
