"""Verifikasi bahwa research/slot_mapping.py identik perilakunya dengan backend produksi.

Kasus uji direplikasi dari oracle yang sudah ada di backend:
  backend-ssb-ai/tests/test_buffet.py
  backend-ssb-ai/tests/test_calibration.py

Bila ada yang GAGAL, salinan di slot_mapping.py-lah yang salah - bukan backend.

Jalankan:  uv run python research/verify_slot_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slot_mapping import (  # noqa: E402
    align_calibration,
    compute_calibration,
    slots_from_calibration,
    slots_from_detections,
)

ANCHOR = (0.0, 0.0, 1000.0, 200.0)
ROWS, COLS = 2, 10

hasil: list[tuple[str, bool, str]] = []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    hasil.append((nama, kondisi, detail))


# --- test_buffet.py ---
summary, slots = slots_from_detections(ANCHOR, [], ROWS, COLS)
cek("frame kosong -> 20 Diangkat", summary["Diangkat"] == 20 and len(slots) == 20)

summary, slots = slots_from_detections(None, [(50.0, 50.0, "penuh", 0.9)], ROWS, COLS)
cek("anchor None -> semua Diangkat", summary["Diangkat"] == 20)

# pemetaan row-major: sudut Nampan 1, 10, 11, 20
trays = [
    (50.0, 50.0, "penuh", 0.9),     # baris 0 kolom 0 -> Nampan 1
    (950.0, 50.0, "habis", 0.9),    # baris 0 kolom 9 -> Nampan 10
    (50.0, 150.0, "sedikit", 0.9),  # baris 1 kolom 0 -> Nampan 11
    (950.0, 150.0, "penuh", 0.9),   # baris 1 kolom 9 -> Nampan 20
]
summary, slots = slots_from_detections(ANCHOR, trays, ROWS, COLS)
peta = {s["id"]: s["status"] for s in slots}
cek("Nampan 1 = Penuh", peta["Nampan 1"] == "Penuh", peta["Nampan 1"])
cek("Nampan 10 = Habis", peta["Nampan 10"] == "Habis", peta["Nampan 10"])
cek("Nampan 11 = Sedikit", peta["Nampan 11"] == "Sedikit", peta["Nampan 11"])
cek("Nampan 20 = Penuh", peta["Nampan 20"] == "Penuh", peta["Nampan 20"])
cek("summary selalu berjumlah 20", sum(summary.values()) == 20, str(sum(summary.values())))

# confidence tertinggi menang dalam satu slot
summary, slots = slots_from_detections(
    ANCHOR, [(50.0, 50.0, "habis", 0.3), (60.0, 60.0, "penuh", 0.95)], ROWS, COLS
)
peta = {s["id"]: s["status"] for s in slots}
cek("confidence tertinggi menang", peta["Nampan 1"] == "Penuh", peta["Nampan 1"])

# centroid di luar anchor diabaikan
summary, _ = slots_from_detections(ANCHOR, [(5000.0, 5000.0, "penuh", 0.9)], ROWS, COLS)
cek("centroid di luar anchor diabaikan", summary["Diangkat"] == 20)

# --- test_calibration.py ---
grid = [(50.0 + c * 100.0, 50.0 + r * 100.0, "penuh", 0.9) for r in range(ROWS) for c in range(COLS)]
cal = compute_calibration(ANCHOR, grid, ROWS, COLS)
cek("compute_calibration menghasilkan 20 slot", len(cal) == 20, str(len(cal)))
cek("nampan bernomor 1..20", [s["nampan"] for s in cal] == list(range(1, 21)))
cek("slot pertama di kiri-atas", cal[0]["rx"] < 0.1 and cal[0]["ry"] < 0.5,
    f"rx={cal[0]['rx']} ry={cal[0]['ry']}")

summary, slots = slots_from_calibration(ANCHOR, [(55.0, 52.0, "penuh", 0.9)], cal, 0.18)
peta = {s["id"]: s["status"] for s in slots}
cek("deteksi dekat menempel ke Nampan 1", peta["Nampan 1"] == "Penuh", peta["Nampan 1"])

summary, _ = slots_from_calibration(ANCHOR, [(5000.0, 5000.0, "penuh", 0.9)], cal, 0.18)
cek("deteksi jauh dibuang", summary["Diangkat"] == len(cal))

# align_calibration memulihkan geseran meja +80px / +10px
geser = [(cx + 80.0, cy + 10.0, s, c) for cx, cy, s, c in grid]
cal_align = align_calibration(ANCHOR, geser, cal)
summary, _ = slots_from_calibration(ANCHOR, geser, cal_align, 0.06)
cek("align_calibration memulihkan geseran (20 termapping @0.06)",
    summary["Diangkat"] == 0, f"Diangkat={summary['Diangkat']}")

# tanpa align, geseran yang sama gagal pada ambang ketat
summary_tanpa, _ = slots_from_calibration(ANCHOR, geser, cal, 0.06)
cek("tanpa align, geseran gagal @0.06 (kontrol)",
    summary_tanpa["Diangkat"] > 0, f"Diangkat={summary_tanpa['Diangkat']}")

# align no-op bila deteksi terlalu sedikit
cal_sedikit = align_calibration(ANCHOR, grid[:2], cal)
cek("align no-op di bawah min_matches", cal_sedikit == cal)

# --- laporan ---
lebar = max(len(n) for n, _, _ in hasil)
gagal = 0
for nama, ok, detail in hasil:
    tanda = "LULUS" if ok else "GAGAL"
    if not ok:
        gagal += 1
    print(f"  [{tanda}] {nama:<{lebar}}" + (f"   -> {detail}" if detail and not ok else ""))

print(f"\n{len(hasil) - gagal}/{len(hasil)} lulus")
if gagal:
    print("\nSalinan di research/slot_mapping.py TIDAK identik dengan perilaku backend.")
    sys.exit(1)
print("Salinan berperilaku identik dengan backend produksi.")
