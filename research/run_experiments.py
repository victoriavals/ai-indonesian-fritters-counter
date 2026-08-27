"""Jalankan seluruh run pelatihan penelitian secara berurutan, tanpa pengawasan.

KENAPA SCRIPT INI ADA
---------------------
Cakupan P0+P1 memerlukan 5 run pelatihan (~13-14 jam GPU). Menjalankannya satu per satu
secara manual mengundang dua kesalahan yang sudah pernah terjadi di project ini:
  1. Salah menunjuk dataset (`dataset_det` yang split-nya bocor, bukan `dataset_det_clean`).
  2. Melatih ulang run yang sudah ada, sehingga angka yang sudah dilaporkan berubah.
Script ini menutup keduanya: dataset diperiksa sebelum mulai, dan run yang sudah punya
`weights/best.pt` DILEWATI secara default.

YANG TIDAK DILAKUKAN SCRIPT INI
-------------------------------
- Tidak menyentuh `backend-ssb-ai/` maupun basis data produksi.
- Tidak melatih ulang `yolo26s_det_v21_clean` (model produksi, seed 0) - run itu sudah ada
  dan sengaja tidak masuk daftar.
- Tidak mengubah config apa pun.

INPUT
-----
- Config di `research/configs/*.yaml`
- Dataset `dataset_det_clean/` (501/80/80)

OUTPUT
------
- `runs/<name>/` per run (weights, results.csv, args.yaml, run_meta.json, plot)
- `research/logs/<name>.log` - stdout+stderr lengkap tiap run
- `research/results/training_runs.csv` - ringkasan: nama, status, durasi, epoch

CARA MENJALANKAN
----------------
    uv run python research/run_experiments.py --list          # lihat rencana, tidak menjalankan
    uv run python research/run_experiments.py --set p0        # 2 run wajib (~4-5 jam)
    uv run python research/run_experiments.py --set p1        # 3 run disarankan (~6-8 jam)
    uv run python research/run_experiments.py --set all       # kelimanya (~13-14 jam)
    uv run python research/run_experiments.py --set all --force   # paksa latih ulang yang sudah ada
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH = REPO_ROOT / "research"
LOG_DIR = RESEARCH / "logs"
RESULTS_DIR = RESEARCH / "results"

# Dataset yang WAJIB dipakai semua run. Diperiksa sebelum apa pun dijalankan, karena
# menunjuk dataset yang salah membuat seluruh tabel perbandingan tidak sah.
REQUIRED_DATA = REPO_ROOT / "dataset_det_clean" / "data.yaml"
EXPECTED_STAMP = "# built_from: v2.1-clean-split.yolo26"

# (config, nama run, prioritas, perkiraan jam) - perkiraan dari 1,744 jam / 134 epoch
# pada RTX 4060 Ti, jadi +-0,78 menit/epoch. Semua angka ini PERKIRAAN, bukan ukuran.
PLAN: list[tuple[str, str, str, str]] = [
    ("research/configs/config-cmp-yolo11s.yaml", "yolo11s_det_v21_clean", "p0", "~2 jam"),
    ("research/configs/config-cmp-yolov8s.yaml", "yolov8s_det_v21_clean", "p0", "~2-2,5 jam"),
    ("research/configs/config-cmp-yolo26n.yaml", "yolo26n_det_v21_clean", "p1", "~1-1,5 jam"),
    ("research/configs/config-seed1-yolo26s.yaml", "yolo26s_det_v21_clean_seed1", "p1", "~1,75 jam"),
    ("research/configs/config-seed2-yolo26s.yaml", "yolo26s_det_v21_clean_seed2", "p1", "~1,75 jam"),
]


def preflight() -> None:
    """Gagal cepat sebelum membakar waktu GPU, bukan gagal di tengah jalan."""
    if not REQUIRED_DATA.is_file():
        sys.exit(
            f"ERROR: dataset tidak ditemukan: {REQUIRED_DATA}\n"
            "Bangun dulu:\n"
            "  uv run python scripts/seg_to_det.py --src v2.1-clean-split.yolo26 "
            "--dst dataset_det_clean"
        )

    # Verifikasi stempel provenance. Ini penjaga terhadap kekeliruan yang paling merusak:
    # melatih di atas partisi Roboflow yang bocor sambil menyangka itu split bersih.
    try:
        first_line = REQUIRED_DATA.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError) as exc:
        sys.exit(f"ERROR: tidak bisa membaca {REQUIRED_DATA}: {exc}")

    if first_line != EXPECTED_STAMP:
        sys.exit(
            f"ERROR: stempel dataset tidak sesuai.\n"
            f"  diharapkan : {EXPECTED_STAMP}\n"
            f"  ditemukan  : {first_line}\n"
            "Ini berarti dataset_det_clean/ TIDAK dibangun dari split group-aware. "
            "Menjalankan pelatihan sekarang akan menghasilkan angka yang tidak sah.\n"
            "Perbaiki dengan:\n"
            "  uv run python scripts/seg_to_det.py --src v2.1-clean-split.yolo26 "
            "--dst dataset_det_clean --clean"
        )

    # Hitung citra per split sebagai pemeriksaan kedua (501/80/80 yang diharapkan).
    counts = {}
    for split in ("train", "valid", "test"):
        d = REPO_ROOT / "dataset_det_clean" / split / "images"
        counts[split] = len(list(d.iterdir())) if d.is_dir() else 0
    print(f"[preflight] dataset_det_clean: {counts} (diharapkan 501/80/80)")
    if (counts["train"], counts["valid"], counts["test"]) != (501, 80, 80):
        print(
            "[preflight] PERINGATAN: jumlah citra berbeda dari 501/80/80. "
            "Jika Anda sengaja menambah data, seluruh perbandingan lama menjadi tidak sebanding."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def already_done(run_name: str) -> bool:
    """Sebuah run dianggap selesai bila `run_meta.json` ada.

    JANGAN memakai keberadaan `weights/best.pt` sebagai penanda selesai. Ultralytics menulis
    best.pt SEJAK EPOCH PERTAMA dan memperbaruinya setiap kali membaik, sehingga run yang
    terhenti di epoch 9 dari 200 pun sudah memiliki best.pt. Memakai berkas itu sebagai
    penanda membuat run setengah terlatih DILEWATI saat rangkaian dijalankan ulang - dan
    naskah akan memuat angka dari model yang belum selesai, tanpa peringatan apa pun.

    `run_meta.json` ditulis oleh scripts/train.py hanya SETELAH `model.train()` kembali,
    sehingga keberadaannya benar-benar berarti pelatihan tuntas. Penanda alternatif yang
    juga hanya muncul di akhir: results.png dan confusion_matrix.png dari validasi final.
    """
    d = REPO_ROOT / "runs" / run_name
    return (d / "run_meta.json").is_file()


def epochs_ran(run_name: str) -> str:
    """Jumlah baris results.csv minus header = jumlah epoch yang benar-benar berjalan."""
    csv_path = REPO_ROOT / "runs" / run_name / "results.csv"
    if not csv_path.is_file():
        return "?"
    try:
        with open(csv_path, encoding="utf-8") as fh:
            return str(max(0, sum(1 for _ in fh) - 1))
    except OSError:
        return "?"


def run_one(config: str, run_name: str) -> tuple[str, float]:
    """Jalankan satu pelatihan. Mengembalikan (status, detik).

    stdout+stderr dialirkan ke berkas log DAN ke terminal, supaya progres tetap terlihat
    tetapi tetap terekam bila sesi terputus.
    """
    log_path = LOG_DIR / f"{run_name}.log"
    cmd = ["uv", "run", "python", "scripts/train.py", "--config", config]
    print(f"\n{'=' * 78}\n[mulai] {run_name}\n  perintah : {' '.join(cmd)}\n  log      : {log_path}\n{'=' * 78}")

    # Proses anak mendeteksi stdout-nya bukan TTY lalu memakai buffering blok, sehingga
    # induk tidak melihat apa pun sampai buffer ~4-8 KB penuh - membuat `tail -f` percuma
    # selama berjam-jam. PYTHONUNBUFFERED memaksa anak menulis tanpa buffer.
    child_env = dict(os.environ, PYTHONUNBUFFERED="1")

    started = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                # Konsol Windows memakai cp1252, sedangkan keluaran Ultralytics memuat
                # karakter Unicode (progress bar, simbol). Menulisnya langsung ke stdout
                # melempar UnicodeEncodeError dan MEMATIKAN seluruh rangkaian pelatihan.
                # Berkas log (UTF-8) tetap menerima baris utuh; cerminan ke terminal
                # dibuat toleran.
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                log_fh.write(line)
                log_fh.flush()   # tanpa ini `tail -f` tetap tertinggal jauh
            code = proc.wait()
    except FileNotFoundError:
        return "GAGAL: 'uv' tidak ditemukan di PATH", time.time() - started
    except KeyboardInterrupt:
        # Jangan menelan Ctrl+C - hentikan seluruh rangkaian.
        proc.terminate()
        raise
    except OSError as exc:
        return f"GAGAL: {exc}", time.time() - started

    elapsed = time.time() - started
    if code != 0:
        return f"GAGAL (exit {code}) - lihat {log_path.name}", elapsed
    if not already_done(run_name):
        return "GAGAL: selesai tanpa error tetapi run_meta.json tidak ada", elapsed
    return "SELESAI", elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Jalankan run pelatihan penelitian secara berurutan.")
    ap.add_argument("--set", default="all", choices=("p0", "p1", "all"), help="subset yang dijalankan")
    ap.add_argument("--list", action="store_true", help="tampilkan rencana lalu keluar")
    ap.add_argument("--force", action="store_true", help="latih ulang walau best.pt sudah ada")
    args = ap.parse_args()

    # Paksa stdout UTF-8 bila didukung, supaya karakter Unicode dari Ultralytics tidak
    # menggagalkan proses pada konsol Windows berkodepage cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    selected = [r for r in PLAN if args.set == "all" or r[2] == args.set]

    print(f"\nRencana ({args.set}) - {len(selected)} run:\n")
    for config, name, prio, est in selected:
        state = "sudah tuntas" if already_done(name) else "belum"
        print(f"  [{prio}] {name:32s} {est:12s} ({state})")
    print("\nPerkiraan waktu adalah ekstrapolasi, bukan ukuran. Semua metrik baru = TO BE MEASURED.")

    if args.list:
        return

    missing = [c for c, _n, _p, _e in selected if not (REPO_ROOT / c).is_file()]
    if missing:
        sys.exit("ERROR: config tidak ditemukan:\n  " + "\n  ".join(missing))

    preflight()

    rows: list[dict[str, str]] = []
    for config, name, prio, _est in selected:
        if already_done(name) and not args.force:
            print(f"\n[lewati] {name} - best.pt sudah ada (pakai --force untuk melatih ulang)")
            rows.append({"run": name, "prioritas": prio, "status": "DILEWATI (sudah tuntas)",
                         "durasi_jam": "-", "epoch": epochs_ran(name)})
            continue
        status, secs = run_one(config, name)
        rows.append({"run": name, "prioritas": prio, "status": status,
                     "durasi_jam": f"{secs / 3600:.3f}", "epoch": epochs_ran(name)})
        print(f"\n[hasil] {name}: {status} ({secs / 3600:.2f} jam)")

    out_csv = RESULTS_DIR / "training_runs.csv"
    try:
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["run", "prioritas", "status", "durasi_jam", "epoch"])
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"PERINGATAN: gagal menulis {out_csv}: {exc}")

    print(f"\n{'=' * 78}\nRINGKASAN\n{'=' * 78}")
    for r in rows:
        print(f"  {r['run']:32s} {r['status']:34s} {r['durasi_jam']:>8s} jam  {r['epoch']:>4s} epoch")
    print(f"\nRingkasan CSV: {out_csv}")

    failed = [r for r in rows if r["status"].startswith("GAGAL")]
    if failed:
        print(f"\n{len(failed)} run GAGAL - periksa research/logs/ sebelum melanjutkan ke evaluasi.")
        sys.exit(1)
    print("\nLangkah berikutnya: evaluasi tiap bobot, lalu compare_models.py untuk membangun T8.")


if __name__ == "__main__":
    main()
