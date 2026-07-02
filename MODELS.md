# Model registry

Which dataset version → which `best.pt` → which metrics. Weights and `runs/` are gitignored,
so this committed file is the durable record (per the project convention). Newest first.

---

## V2 — `yolo26s_det_v2` (2026-07-02) — **trained, NOT yet deployed**

- **Dataset:** `v2-gorengan-counter.yolo26` (Roboflow project `v2-gorengan-counter`), `nc=5`:
  `['habis', 'hampir habis', 'meja', 'penuh', 'sedikit']` — fill state by volume
  (0% / 1–25% / table / 51–100% / 26–50%). Splits: 169 train / 12 valid / 10 test.
- **Base / recipe:** `yolo26s.pt`, 200 epochs (best @ **epoch 175**), imgsz 640, batch 16,
  cos_lr, close_mosaic 10, seed 0. Full args in `runs/yolo26s_det_v2/{run_meta.json,args.yaml}`.
- **Weights:** `runs/yolo26s_det_v2/weights/best.pt` (YOLO26s, 9.47M params, 20.5 GFLOPs).
- **Training time:** ~0.61 h on RTX 4060 Ti (8 GB).

**Metrics — TEST split** (10 img / 154 inst — *small, noisy*):

| Class | P | R | mAP50 | mAP50-95 |
|---|--:|--:|--:|--:|
| **all** | 0.937 | 0.945 | **0.949** | 0.859 |
| habis | 1.000 | 0.962 | 0.995 | 0.915 |
| hampir habis | 0.879 | 0.979 | 0.927 | 0.838 |
| meja | 0.939 | 1.000 | 0.995 | 0.985 |
| penuh | 0.956 | 0.930 | 0.932 | 0.790 |
| sedikit | 0.913 | 0.854 | 0.898 | 0.764 |

**Metrics — VALID split** (12 img / 177 inst): all → P 0.967 / R 0.981 / mAP50 **0.977** / mAP50-95 0.881.

**Caveats:**
- Val/test splits are tiny → metrics are noisy. `habis` has only 5 test instances → its "perfect"
  numbers are not statistically meaningful.
- `sedikit` is the weakest class (test R=0.854, mAP50=0.898) — likely confused with its neighbours
  `hampir habis` / `penuh`. Check `runs/yolo26s_det_v2/confusion_matrix.png`.
- `meja` anchor detection is near-perfect (mAP50 0.995) → the backend's meja-anchored slot mapping
  will stay reliable.
- **Real validation is on-site CCTV frames**, not this held-out set.

**Deploy status:** the backend + frontend code were **updated to the V2 5-state scheme**
(2026-07-02) — states Penuh/Sedikit/Hampir Habis/Habis/Diangkat; alerts fire early on
"Hampir Habis" too. 60 backend tests pass; frontend builds clean. The **live model + Postgres
are still V1** — do the cutover as one coordinated set:
1. `cd backend-ssb-ai && uv run python scripts/migrate_v2_status.py` (Postgres: kosong→habis, +hampir_habis)
2. copy `runs/yolo26s_det_v2/weights/best.pt` → `backend-ssb-ai/models/best.pt`
3. `cd frontend-gorengan-ssb && npm run build`
4. `service.ps1 restart` (or restart the backend + frontend)

---

## V1 — `nc=4` (2026-06-24) — **deployed**

- **Dataset:** `gorengan-conter.yolo26` (Roboflow project `gorengan-conter`), `nc=4`:
  `['kosong', 'meja', 'penuh', 'sedikit']`. Splits: 163 train / 16 valid / 13 test. *(Dataset deleted 2026-07-02 when V2 replaced it; still recoverable from Roboflow.)*
- **Weights:** currently live at `backend-ssb-ai/models/best.pt`.
- **Metrics:** mAP50 ≈ **0.949** on the test set.
