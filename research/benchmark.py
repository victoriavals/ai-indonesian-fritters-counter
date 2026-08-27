"""Ukur latensi inferensi, ukuran model, dan memori - untuk Tabel T9 paper.

KENAPA SCRIPT INI ADA
---------------------
Angka 24 ms (GPU) / 196 ms (CPU) / 178 ms (median 8 thread) / RSS 656 MB yang dikutip di
DEPLOY-VPS-CPU.md dan CLAUDE.md TIDAK punya script pendukung - semuanya hanya klaim prosa.
Untuk paper, angka tanpa metode yang dapat dijalankan ulang adalah masalah. Script ini
mereplikasi metode yang terdokumentasi supaya hasilnya sebanding dengan angka lama:

  - `torch.set_num_threads(8)` saat mengukur CPU
  - `imgsz=640`
  - median dari 10 run (bukan rata-rata - median tahan terhadap pencilan)
  - batas pengukuran `perf_counter` PERSIS di sekitar `model.predict(...)`, mengikuti
    backend-ssb-ai/app/services/yolo.py:74-80, sehingga tidak memasukkan waktu dekode citra

Warm-up dijalankan lebih dulu dan TIDAK dihitung: run pertama memuat kernel CUDA/cuDNN dan
akan mencemari median.

INPUT
-----
- Satu atau lebih berkas bobot (.pt)
- Satu citra uji. Default: citra pertama dari dataset_det_clean/test/images
  (frame CCTV asli 768x432, sama seperti kondisi produksi)

OUTPUT
------
- research/results/benchmark.csv  -> dipakai compare_models.py --benchmark untuk T8/T9
- ringkasan ke stdout

CARA MENJALANKAN
----------------
    # model produksi saja
    uv run python research/benchmark.py -w runs/yolo26s_det_v21_clean/weights/best.pt -l YOLO26s

    # beberapa model sekaligus (setelah run pembanding selesai)
    uv run python research/benchmark.py \\
        -w runs/yolo26s_det_v21_clean/weights/best.pt \\
        -w runs/yolo11s_det_v21_clean/weights/best.pt \\
        -l YOLO26s -l YOLOv11s

    # hanya CPU (mis. mensimulasikan server produksi tanpa GPU)
    uv run python research/benchmark.py -w <bobot> -l <label> --devices cpu

CATATAN KEJUJURAN
-----------------
Angka CPU di mesin ini (RTX 4060 Ti + CPU desktop) TIDAK sama dengan angka di server produksi
(Xeon Silver 4410Y, 8 core, tanpa GPU). Laporkan spesifikasi mesin pengukur di paper, dan
jangan menyamakan keduanya.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "research" / "results"

DEFAULT_IMGSZ = 640
DEFAULT_RUNS = 10        # median dari 10, seperti metode terdokumentasi
DEFAULT_WARMUP = 3
CPU_THREADS = 8          # torch.set_num_threads(8), seperti metode terdokumentasi


def pick_default_image() -> Path:
    """Ambil citra uji pertama dari split test bersih."""
    d = REPO_ROOT / "dataset_det_clean" / "test" / "images"
    if d.is_dir():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                return p
    sys.exit(
        "ERROR: tidak menemukan citra uji default di dataset_det_clean/test/images.\n"
        "Sebutkan citra secara eksplisit dengan --image <path>."
    )


def model_stats(model) -> tuple[str, str]:
    """(jumlah parameter, GFLOPs) dari model Ultralytics; '-' bila tidak tersedia.

    PENTING - model DI-FUSE lebih dulu. Ultralytics melaporkan ringkasan model dalam
    keadaan fused (BatchNorm dilipat ke konvolusi), dan itulah angka yang sudah tercatat
    di MODELS.md serta logs/fair-eval.log: "YOLO26s summary (fused): 122 layers,
    9,467,115 parameters, 20.5 GFLOPs". Tanpa fuse, hitungannya ~9,95 M / 22,5 GFLOPs -
    berbeda, dan paper akan memuat dua angka parameter yang bertentangan.
    Fuse juga mencerminkan model yang benar-benar dipakai saat inferensi.
    """
    params = gflops = "-"
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        try:
            model.fuse()  # idempotent; aman dipanggil walau sudah fused
        except Exception as exc:  # noqa: BLE001 - lanjutkan dengan angka unfused, tapi beritahu
            print(f"[warn] fuse gagal ({exc}) - angka params/GFLOPs akan versi UNFUSED")

        params = f"{get_num_params(model.model):,}"
        f = get_flops(model.model, imgsz=DEFAULT_IMGSZ)
        gflops = f"{f:.1f}" if f else "-"
    except Exception as exc:  # noqa: BLE001 - informasi tambahan, jangan gagalkan benchmark
        print(f"[warn] gagal membaca params/GFLOPs: {exc}")
    return params, gflops


def bench_one(weights: Path, image: Path, device: str, runs: int, warmup: int) -> dict[str, str]:
    """Ukur satu bobot pada satu device. Mengembalikan baris hasil."""
    import torch
    from PIL import Image
    from ultralytics import YOLO

    if device == "cpu":
        torch.set_num_threads(CPU_THREADS)
    elif not torch.cuda.is_available():
        print(f"[skip] device 'cuda' diminta tetapi tidak tersedia - {weights.name} dilewati")
        return {}

    img = Image.open(image).convert("RGB")
    model = YOLO(str(weights))
    params, gflops = model_stats(model)

    # Warm-up: TIDAK dihitung. Run pertama memuat kernel dan akan mencemari median.
    for _ in range(warmup):
        model.predict(source=img, imgsz=DEFAULT_IMGSZ, device=device, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=DEFAULT_IMGSZ, device=device, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()   # tanpa ini, CUDA asinkron membuat waktu terlihat mustahil cepat
        timings.append((time.perf_counter() - t0) * 1000.0)

    # RSS diukur SETELAH model dimuat + inferensi, sesuai catatan metode lama.
    rss_mb = "-"
    try:
        import psutil

        rss_mb = f"{psutil.Process().memory_info().rss / 1024 ** 2:.0f}"
    except ImportError:
        pass

    return {
        "params": params,
        "gflops": gflops,
        "median_ms": f"{statistics.median(timings):.1f}",
        "mean_ms": f"{statistics.mean(timings):.1f}",
        "min_ms": f"{min(timings):.1f}",
        "max_ms": f"{max(timings):.1f}",
        "stdev_ms": f"{statistics.stdev(timings):.1f}" if len(timings) > 1 else "0.0",
        "fps_median": f"{1000.0 / statistics.median(timings):.1f}",
        "rss_mb": rss_mb,
        "size_mb": f"{weights.stat().st_size / 1024 ** 2:.1f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark latensi & sumber daya model (T9).")
    ap.add_argument("-w", "--weights", action="append", required=True, help="berkas bobot (boleh berulang)")
    ap.add_argument("-l", "--label", action="append", help="label model (boleh berulang, sejajar dengan -w)")
    ap.add_argument("--image", help="citra uji (default: citra pertama split test bersih)")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"], choices=["cuda", "cpu"])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"jumlah run terukur (default {DEFAULT_RUNS})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"run pemanasan (default {DEFAULT_WARMUP})")
    args = ap.parse_args()

    weights = [(REPO_ROOT / w).resolve() if not Path(w).is_absolute() else Path(w) for w in args.weights]
    for w in weights:
        if not w.is_file():
            sys.exit(f"ERROR: bobot tidak ditemukan: {w}")
    labels = args.label or [w.parent.parent.name for w in weights]
    if len(labels) != len(weights):
        sys.exit(f"ERROR: jumlah label ({len(labels)}) tidak sama dengan bobot ({len(weights)})")

    image = Path(args.image).resolve() if args.image else pick_default_image()
    if not image.is_file():
        sys.exit(f"ERROR: citra uji tidak ditemukan: {image}")

    from PIL import Image

    with Image.open(image) as im:
        w_px, h_px = im.size

    print("METODE (mereplikasi DEPLOY-VPS-CPU.md supaya sebanding dengan angka lama):")
    print(f"  citra uji     : {image.name}  ({w_px}x{h_px})")
    print(f"  imgsz         : {DEFAULT_IMGSZ}")
    print(f"  run terukur   : {args.runs} (dilaporkan median)")
    print(f"  warm-up       : {args.warmup} (tidak dihitung)")
    print(f"  thread CPU    : torch.set_num_threads({CPU_THREADS})")
    print(f"  batas ukur    : perf_counter di sekitar model.predict()\n")

    rows: list[dict[str, str]] = []
    for w, label in zip(weights, labels):
        per_device: dict[str, dict[str, str]] = {}
        for dev in args.devices:
            print(f"[ukur] {label} @ {dev} ...", flush=True)
            try:
                res = bench_one(w, image, dev, args.runs, args.warmup)
            except Exception as exc:  # noqa: BLE001 - satu device gagal jangan menggagalkan semua
                print(f"[error] {label} @ {dev}: {exc}")
                res = {}
            if res:
                per_device[dev] = res
                print(f"        median {res['median_ms']} ms  "
                      f"(min {res['min_ms']} / max {res['max_ms']} / sd {res['stdev_ms']})  "
                      f"{res['fps_median']} fps")

        base = per_device.get("cuda") or per_device.get("cpu") or {}
        rows.append({
            "model": label,
            "bobot": str(w.relative_to(REPO_ROOT)) if REPO_ROOT in w.parents else str(w),
            "params": base.get("params", "-"),
            "gflops": base.get("gflops", "-"),
            "size_mb": base.get("size_mb", "-"),
            "gpu_ms_median": per_device.get("cuda", {}).get("median_ms", "-"),
            "gpu_fps": per_device.get("cuda", {}).get("fps_median", "-"),
            "cpu_ms_median": per_device.get("cpu", {}).get("median_ms", "-"),
            "cpu_fps": per_device.get("cpu", {}).get("fps_median", "-"),
            "cpu_rss_mb": per_device.get("cpu", {}).get("rss_mb", "-"),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark.csv"
    try:
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w_csv.writeheader()
            w_csv.writerows(rows)
    except (OSError, IndexError) as exc:
        sys.exit(f"ERROR: gagal menulis {out}: {exc}")

    print(f"\n{'model':14s}{'params':>13s}{'GFLOPs':>9s}{'GPU ms':>9s}{'CPU ms':>9s}{'RSS MB':>9s}")
    print("-" * 63)
    for r in rows:
        print(f"{r['model']:14s}{r['params']:>13s}{r['gflops']:>9s}"
              f"{r['gpu_ms_median']:>9s}{r['cpu_ms_median']:>9s}{r['cpu_rss_mb']:>9s}")

    print(f"\nCSV: {out}")
    print("\nCatatan untuk paper: angka CPU di mesin ini BUKAN angka server produksi"
          " (Xeon Silver 4410Y, tanpa GPU). Sebutkan spesifikasi mesin pengukur di naskah.")


if __name__ == "__main__":
    main()
