"""Pemetaan slot nampan - SALINAN VERBATIM dari backend, tanpa dependensi produksi.

KENAPA DISALIN, BUKAN DIIMPOR
-----------------------------
`backend-ssb-ai/app/services/buffet.py` mengimpor `app.services.calibration`, yang mengimpor
`app.database`. Modul itu membangun engine SQLAlchemy **saat import**, dan `.env` backend
menunjuk basis data **PRODUKSI** (Railway Postgres). Jadi satu baris `import app.services.buffet`
di skrip penelitian akan mengarah ke basis data produksi.

Fungsi-fungsi di bawah ini **murni** - hanya membaca argumennya, tanpa menyentuh basis data,
model, maupun cache global. Karena itu aman disalin utuh.

CARA BERKAS INI DIBUAT
----------------------
Diekstrak secara PROGRAMATIS dari sumbernya (bukan diketik ulang), agar hasilnya identik bit
per bit dengan yang berjalan di produksi. Regenerasi:

    bash research/regen_slot_mapping.sh

Sumber:
  - backend-ssb-ai/app/services/buffet.py       baris 32-41, 66-68, 71-149
  - backend-ssb-ai/app/services/calibration.py  baris 35-61, 64-67, 70-118

YANG SENGAJA TIDAK DISALIN
--------------------------
- `_last_anchor` (cache anchor per cabang) - state global mutable, tidak relevan untuk
  evaluasi frame tunggal.
- `analyze_frame()` - memerlukan model YOLO dan pembacaan kalibrasi dari basis data.
- `save_calibration/get_calibration/reload_calibration` - semuanya menyentuh basis data.

ORACLE KEBENARAN
----------------
Perilaku salinan ini harus sama dengan test yang sudah ada di backend:
  tests/test_buffet.py, tests/test_calibration.py
Bila hasilnya berbeda, salinan inilah yang salah.
"""

from __future__ import annotations


STATUS_DISPLAY = {
    "penuh": "Penuh",
    "sedikit": "Sedikit",
    "hampir habis": "Hampir Habis",
    "habis": "Habis",
    "diangkat": "Diangkat",
}
# Fill classes the model emits per tray (everything except the 'meja' anchor).
TRAY_CLASSES = {"penuh", "sedikit", "hampir habis", "habis"}
ANCHOR_CLASS = "meja"

def _centroid(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def slots_from_detections(
    anchor: tuple[float, float, float, float] | None,
    trays: list[tuple[float, float, str, float]],
    rows: int,
    cols: int,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Pure spatial mapping (no model) — easy to unit-test.

    anchor: the 'meja' box (x1,y1,x2,y2) or None.
    trays:  list of (cx, cy, status_lower, conf).
    Returns (summary, slots). Slots are row-major; empty slots become "Diangkat".
    """
    n_slots = rows * cols
    slot_status = ["diangkat"] * n_slots
    slot_conf = [0.0] * n_slots

    if anchor is not None:
        ax1, ay1, ax2, ay2 = anchor
        cell_w = (ax2 - ax1) / cols
        cell_h = (ay2 - ay1) / rows
        if cell_w > 0 and cell_h > 0:
            for cx, cy, status, conf in trays:
                col = int((cx - ax1) / cell_w)
                row = int((cy - ay1) / cell_h)
                if 0 <= col < cols and 0 <= row < rows:
                    idx = row * cols + col
                    if conf > slot_conf[idx]:  # keep the most confident detection per slot
                        slot_status[idx] = status
                        slot_conf[idx] = conf

    summary = {"Penuh": 0, "Sedikit": 0, "Hampir Habis": 0, "Habis": 0, "Diangkat": 0}
    slots: list[dict[str, str]] = []
    for i, status in enumerate(slot_status):
        disp = STATUS_DISPLAY[status]
        summary[disp] += 1
        slots.append({"id": f"Nampan {i + 1}", "status": disp})
    return summary, slots


def slots_from_calibration(
    anchor: tuple[float, float, float, float],
    trays: list[tuple[float, float, str, float]],
    calibration: list[dict],
    max_dist: float,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Map centroids to the NEAREST calibrated slot (relative to the anchor).

    Accurate for fixed cameras: calibration captures the real (perspective-distorted)
    slot positions. A detection only snaps if within `max_dist` (relative) of a slot;
    slots with no detection become 'Diangkat'.
    """
    ax1, ay1, ax2, ay2 = anchor
    aw = ax2 - ax1
    ah = ay2 - ay1
    n = len(calibration)
    slot_status = ["diangkat"] * n
    slot_conf = [0.0] * n
    max_d2 = max_dist * max_dist

    if aw > 0 and ah > 0:
        for cx, cy, status, conf in trays:
            rx = (cx - ax1) / aw
            ry = (cy - ay1) / ah
            best_i, best_d2 = -1, max_d2
            for i, s in enumerate(calibration):
                d2 = (rx - s["rx"]) ** 2 + (ry - s["ry"]) ** 2
                if d2 < best_d2:
                    best_d2, best_i = d2, i
            if best_i >= 0 and conf > slot_conf[best_i]:
                slot_status[best_i] = status
                slot_conf[best_i] = conf

    summary = {"Penuh": 0, "Sedikit": 0, "Hampir Habis": 0, "Habis": 0, "Diangkat": 0}
    slots: list[dict[str, str]] = []
    for i, status in enumerate(slot_status):
        disp = STATUS_DISPLAY[status]
        summary[disp] += 1
        slots.append({"id": f"Nampan {calibration[i]['nampan']}", "status": disp})
    return summary, slots

def compute_calibration(
    anchor: tuple[float, float, float, float],
    trays: list[tuple[float, float, str, float]],
    rows: int,
    cols: int,
) -> list[dict]:
    """Order tray centroids row-major and return slot centers relative to the anchor.

    Expects ~rows*cols trays present in the reference frame. Splits by Y into rows
    (top→bottom), then sorts each row by X (left→right) → Nampan 1..N.
    """
    ax1, ay1, ax2, ay2 = anchor
    aw = ax2 - ax1
    ah = ay2 - ay1
    rel = [((cx - ax1) / aw, (cy - ay1) / ah) for cx, cy, _s, _c in trays]
    rel.sort(key=lambda p: p[1])  # top → bottom

    n = len(rel)
    per_row = max(1, round(n / rows))
    slots: list[dict] = []
    for r in range(rows):
        start = r * per_row
        end = n if r == rows - 1 else min(n, (r + 1) * per_row)
        row = sorted(rel[start:end], key=lambda p: p[0])  # left → right
        for rx, ry in row:
            slots.append({"nampan": len(slots) + 1, "rx": round(rx, 5), "ry": round(ry, 5)})
    return slots

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def align_calibration(
    anchor: tuple[float, float, float, float],
    trays: list[tuple[float, float, str, float]],
    calibration: list[dict],
    match_thresh: float = 0.14,
    min_matches: int = 4,
    iters: int = 3,
) -> list[dict]:
    """Align the calibrated template to the actually-detected trays for THIS frame.

    Corrects table movement with a TRANSLATION-ONLY shift (robust median of match
    residuals) — deliberately not scale/affine, which tends to over-stretch and
    push the edge slots off their trays. Returns adjusted slot positions relative
    to the anchor; falls back to the input calibration if too few detections.
    """
    ax1, ay1, ax2, ay2 = anchor
    aw, ah = ax2 - ax1, ay2 - ay1
    if aw <= 0 or ah <= 0 or len(trays) < min_matches:
        return calibration

    dets = [((cx - ax1) / aw, (cy - ay1) / ah) for cx, cy, _s, _c in trays]

    # Step 0: bulk translation (detected centroid -> template centroid).
    dcx = sum(p[0] for p in dets) / len(dets)
    dcy = sum(p[1] for p in dets) / len(dets)
    scx = sum(s["rx"] for s in calibration) / len(calibration)
    scy = sum(s["ry"] for s in calibration) / len(calibration)
    cal = [{"nampan": s["nampan"], "rx": s["rx"] + (dcx - scx), "ry": s["ry"] + (dcy - scy)} for s in calibration]

    # Refine: nearest-match, then shift by the median residual (translation only).
    t2 = match_thresh * match_thresh
    for _ in range(iters):
        rxs, rys = [], []
        for drx, dry in dets:
            best_i, best_d2 = -1, t2
            for i, sl in enumerate(cal):
                dd = (drx - sl["rx"]) ** 2 + (dry - sl["ry"]) ** 2
                if dd < best_d2:
                    best_d2, best_i = dd, i
            if best_i >= 0:
                rxs.append(drx - cal[best_i]["rx"])
                rys.append(dry - cal[best_i]["ry"])
        if len(rxs) < min_matches:
            break
        mx, my = _median(rxs), _median(rys)
        if abs(mx) < 1e-4 and abs(my) < 1e-4:
            break
        cal = [{"nampan": s["nampan"], "rx": s["rx"] + mx, "ry": s["ry"] + my} for s in cal]
    return cal
