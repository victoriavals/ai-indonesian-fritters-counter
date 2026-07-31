# Model registry

Which dataset version → which `best.pt` → which metrics. Weights and `runs/` are gitignored,
so this committed file is the durable record (per the project convention). Newest first.

---

## V2.1 clean-split — `yolo26s_det_v21_clean` (2026-07-30) — **trained, NOT deployed; USE THESE NUMBERS**

**This is the only V2.1 run whose metrics are trustworthy.** Same data, same hyperparameters, same
seed as `yolo26s_det_v21` below — the *only* difference is the train/valid/test partition. Its
sibling's headline 0.988 is inflated by test-set leakage; this run's 0.941 is honest.

- **Dataset:** `v2.1-clean-split.yolo26` → `dataset_det_clean` — the same 661 V2.1 images
  regrouped by `scripts/make_clean_split.py`: frames clustered by visual similarity
  (thumbnail-MAE < 12 → 182 scene groups), then **whole groups** assigned to splits so no
  near-duplicate can straddle two of them. Splits: 501 train / 80 valid / 80 test.
  32.4% of images changed split versus the Roboflow partition.
- **Leakage, measured:** nearest-train-neighbour MAE — test min **12.16** / median 16.27,
  valid min 12.17 / median 14.08; **0% of eval images under MAE 5** (baseline: 36%).
- **Base / recipe:** identical to the baseline (`yolo26s.pt`, imgsz 640, batch 16, cos_lr,
  close_mosaic 10, seed 0, workers 0, patience 50). Config: `config-v21-clean.yaml`.
- **Ran 134/200 epochs** — early-stopped; **best epoch 84** (by fitness `0.1·mAP50 + 0.9·mAP50-95`).
- **Weights:** `runs/yolo26s_det_v21_clean/weights/best.pt`. **Training time:** 1.744 h.

**Metrics — TEST split** (80 img / 1,454 inst):

| Class | P | R | mAP50 | mAP50-95 |
|---|--:|--:|--:|--:|
| **all** | 0.954 | 0.896 | **0.941** | 0.862 |
| habis | 0.984 | 0.857 | 0.874 | 0.704 |
| hampir habis | 0.968 | 0.865 | 0.949 | 0.866 |
| meja | 1.000 | 0.992 | 0.995 | 0.993 |
| penuh | 0.988 | 0.917 | 0.986 | 0.910 |
| sedikit | 0.831 | 0.847 | 0.902 | 0.834 |

**Metrics — VALID split** (80 img / 1,619 inst): all → P 0.821 / R 0.936 / mAP50 **0.938** / mAP50-95 0.873.

### Baseline vs clean — how much the leaky split was flattering us (TEST split, both runs)

| Metric | `v21` (leaky) | `v21_clean` (honest) | Δ |
|---|--:|--:|--:|
| mAP50 (all) | 0.988 | **0.941** | −0.047 |
| mAP50-95 (all) | 0.943 | **0.862** | −0.081 |
| `sedikit` mAP50 | 0.971 | **0.902** | −0.069 |
| `habis` mAP50 | 0.995 | **0.874** | −0.121 |
| `meja` mAP50 | 0.993 | 0.995 | +0.002 |

Read this as: **the real model is ~0.05 mAP50 / ~0.08 mAP50-95 worse than the leaky split
claimed**, and the gap is concentrated exactly where it hurts — the fill-level classes. `meja`
is unaffected, which is good news: the backend's meja-anchored slot mapping stays reliable.

**What this run tells us (and what it does not):**
- **Plateaus early.** Fitness kept wandering rather than climbing after ~epoch 80, and per-epoch
  mAP50 swung between 0.784 and 0.954 (stdev 0.050 after epoch 60) — versus the baseline's smooth
  curve. That smoothness was an artefact of evaluating on near-copies of training data. The
  volatility here is the honest signal: **the ceiling is data diversity, not training length**.
  More epochs / lr tweaks / imgsz changes will not break through it.
- **Field validation needs several sessions.** With per-epoch stdev ≈ 0.05 on unseen scenes, a
  single short on-site measurement can be off by that much by chance alone.
- **`sedikit` is the weakest class, again** (mAP50 0.902, P 0.831) — third run in a row. The
  25%/50% fill boundaries are inherently fuzzy; this is a labelling-definition problem as much
  as a model one. Check `runs/yolo26s_det_v21_clean/confusion_matrix.png`.
- **`habis` numbers here are NOT reliable — only 7 test instances.** Its 0.874 could move a lot
  either way. This is the class the entire alert feature exists for, and it is still 2.2% of the
  dataset (269/12,200) against `penuh`'s 42%.

### Recommended next actions, in impact order

1. **Collect and label more `habis` frames** (and more `sedikit`/`hampir habis` boundary cases).
   No hyperparameter substitutes for this. Target enough that the *test* split alone holds
   50+ `habis` instances, or its metric stays meaningless.
2. **Add a second camera/branch** before trusting multi-branch rollout. All 661 frames are one
   camera at `cijantung-kavelri`, 2026-06-20..22, 02:57–20:06.
3. **Then** consider `imgsz: 960` (batch 8 on 8 GB) — the fill-level boundary is the one thing
   in-tray texture detail should help, but only once 1–2 remove the bigger bottlenecks.
4. Re-run `make_clean_split.py` after any dataset addition; never trust a Roboflow random split
   for this dataset.

**Deploy status:** not deployed. `backend-ssb-ai/models/best.pt` still holds the V2 weights
(verified by hash 2026-07-30). If you deploy this, expect roughly the numbers above — not 0.988.

### 🔴 Head-to-head vs the LIVE V2 model on an identical, fully fair test set

Both models evaluated on **the same 73 images / 1,421 instances** — the clean test split minus
**every** frame V2 ever trained *or* validated on (7 removed: 4445, 4491, 4492, 4501–4504). So
neither model has seen any of it. Same class scheme (`nc=5`, same order), same imgsz, `workers=0`.
Built by the snippet recorded in this repo's history; eval log: `logs/fair-eval.log`.

| Metric | **V2 (LIVE today)** | **V2.1 clean** | Δ |
|---|--:|--:|--:|
| mAP50 (all) | **0.629** | **0.942** | **+0.313** |
| mAP50-95 (all) | 0.542 | 0.863 | +0.321 |
| Precision | 0.679 | 0.954 | +0.275 |
| Recall | 0.676 | 0.898 | +0.222 |

Per class (mAP50), with instance counts so reliability is visible:

| Class | inst | V2 (live) | V2.1 clean | Δ | Reliable? |
|---|--:|--:|--:|--:|---|
| `habis` | 7 | **0.166** | 0.875 | +0.709 | ✗ too few — direction only |
| `sedikit` | 222 | **0.377** | 0.905 | +0.528 | ✓ |
| `hampir habis` | 338 | 0.769 | 0.949 | +0.181 | ✓ |
| `penuh` | 781 | 0.838 | 0.986 | +0.148 | ✓ |
| `meja` | 73 | 0.995 | 0.995 | 0.000 | ✓ |

**What this means:** the live V2 model scores **0.629 mAP50 on scenes it has not seen** — not the
0.949 its own section below records (that figure came from a leaky 10-image test split). On
well-sampled classes the gap is unambiguous: `sedikit` 0.377 (precision 0.303 — it over-fires
badly), `penuh` recall 0.730 (it misses ~27% of full trays). `habis` recall is 0.143 — it found
**1 of 7** empty trays; with only 7 instances treat the number as indicative, not exact, but the
direction matters because `habis` is the entire reason the alert feature exists.

`meja` is **identical (0.995)** in both. That is the control that makes the rest credible: the
detector backbone still works and the table anchor is unaffected — what degraded is specifically
fill-level discrimination on unfamiliar scenes.

**Honest caveat:** V2.1 is a *re-labelled and expanded* export, so some of V2's deficit could be
labelling-convention drift between the two exports rather than pure model weakness (especially at
the fuzzy `sedikit`/`hampir habis` boundary). But drift alone cannot plausibly account for a
0.313 mAP50 gap while `meja` stays pixel-identical, nor for `penuh` losing 27% recall.

**Conclusion: deploying V2.1 clean is a large, real improvement over what is live**, and the
current production numbers in the V2 section below should not be trusted. See the deploy
checklist in the V2 section for the mechanics (Postgres already migrated — only steps 2–4 apply).

---

## V2.1 baseline — `yolo26s_det_v21` (2026-07-30) — **superseded by the clean-split run above; metrics INFLATED**

- **Dataset:** `v2.1-gorengan-counter.yolo26` (Roboflow workspace `sahabats-workspace`, project
  `v2-gorengan-counter-pcnkp` — a **clone** of the primary `naufalfirdaus` workspace), `nc=5`,
  same class order as V2. Splits **as Roboflow shipped them**: 566 train / 48 valid / 47 test
  (661 images, 12,200 instances). Converted with `seg_to_det.py` — 0 degenerate boxes dropped.
- **Base / recipe:** `yolo26s.pt`, imgsz 640, batch 16, cos_lr, close_mosaic 10, seed 0,
  workers 0, patience 50. Full args in `runs/yolo26s_det_v21/{run_meta.json,args.yaml}`.
- **Ran 126/200 epochs** — early-stopped (no fitness improvement for 50 epochs).
  **Best epoch 76**, chosen by Ultralytics *fitness* (`0.1·mAP50 + 0.9·mAP50-95`), not by mAP50
  alone — so it optimises box-localisation quality, which is what the backend's meja-anchored
  slot mapping depends on.
- **Weights:** `runs/yolo26s_det_v21/weights/best.pt` (19.4 MB).
- **Training time:** 1.74 h on RTX 4060 Ti (8 GB), sharing the GPU with the live backend.

**Metrics — TEST split** (47 img / 840 inst):

| Class | P | R | mAP50 | mAP50-95 |
|---|--:|--:|--:|--:|
| **all** | 0.966 | 0.965 | **0.988** | 0.943 |
| habis | 1.000 | 0.972 | 0.995 | 0.920 |
| hampir habis | 0.969 | 0.985 | 0.990 | 0.948 |
| meja | 0.966 | 0.957 | 0.993 | 0.989 |
| penuh | 0.966 | 0.978 | 0.993 | 0.940 |
| sedikit | 0.932 | 0.933 | 0.971 | 0.918 |

**Metrics — VALID split** (48 img / 826 inst): all → P 0.983 / R 0.955 / mAP50 **0.989** / mAP50-95 0.929.

**⚠️ CAVEAT — these numbers are inflated; do NOT read them as production readiness:**
- The shipped split is **randomly partitioned over frames**, but the frames are screenshots
  grabbed minutes apart from the same CCTV recording. Measured leakage: **36% of test images
  (17/47) have a train twin at 64×64 grayscale pixel-MAE < 5** (random-pair baseline ≈ 38);
  the closest pair is **1 min 54 s apart in the same scene**, one in train, one in test.
  The model is therefore scored largely on near-copies of what it memorised.
- Consequence: **this run is NOT comparable to V2's 0.949** either — V2 used the same
  random-split methodology, so both headline numbers carry unknown, differing inflation.
  Its real value is as a *baseline for measuring that inflation* against the clean-split run.
- Use `yolo26s_det_v21_clean` (group-aware split, see `scripts/make_clean_split.py`) for any
  honest accuracy claim.
- `habis` remains statistically thin: only **12 test instances** (269 of 12,200 total = 2.2%),
  against `penuh`'s 5,135 — a 19:1 imbalance on the class the whole alert feature exists for.
  Its perfect-looking P=1.000 is not meaningful. **Collecting more `habis` frames is the single
  highest-value dataset action**, and no hyperparameter change substitutes for it.
- `sedikit` is again the weakest class (mAP50 0.971, R 0.933) — same as V2. Expected: the
  25%/50% fill boundaries are inherently fuzzy. See `runs/yolo26s_det_v21/confusion_matrix.png`.
- Still **one camera, one branch** (`cijantung-kavelri`), 3 days (2026-06-20..22), times
  02:57–20:06. Real accuracy on a *different* branch's camera angle is unmeasured.

**Deploy status:** not deployed. `backend-ssb-ai/models/best.pt` still holds the V2 weights.

---

## V2 — `yolo26s_det_v2` (2026-07-02) — **DEPLOYED (currently live)**

- **Dataset:** `v2-gorengan-counter.yolo26` (Roboflow project `v2-gorengan-counter`), `nc=5`:
  `['habis', 'hampir habis', 'meja', 'penuh', 'sedikit']` — fill state by volume
  (0% / 1–25% / table / 51–100% / 26–50%). Splits: 169 train / 12 valid / 10 test.
- **Base / recipe:** `yolo26s.pt`, 200 epochs (best @ **epoch 175**), imgsz 640, batch 16,
  cos_lr, close_mosaic 10, seed 0. Full args in `runs/yolo26s_det_v2/{run_meta.json,args.yaml}`.
- **Weights:** `runs/yolo26s_det_v2/weights/best.pt` (YOLO26s, 9.47M params, 20.5 GFLOPs).
- **Training time:** ~0.61 h on RTX 4060 Ti (8 GB).

> ⚠️ **The metrics below are NOT representative of this model's real behaviour.** Measured on a
> fair, unseen 73-image set (see the V2.1-clean section above) this model scores **mAP50 0.629**,
> not 0.949 — with `sedikit` at 0.377 and `habis` recall at 0.143. The 10-image test split used
> here shares near-duplicate frames with its own training data. Keep the table for provenance,
> but quote the fair-comparison figures instead.

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

**Deploy status: LIVE.** The cutover was completed on 2026-07-02 — Postgres migrated
(`kosong`→`habis`, `hampir_habis` added), weights copied, frontend rebuilt, stack restarted.
Backend + frontend run the V2 5-state scheme (Penuh/Sedikit/Hampir Habis/Habis/Diangkat, with
alerts firing early on "Hampir Habis"). V1 weights kept as `models/best_v1_backup.pt` for rollback.

Verified 2026-07-30: `sha256(backend-ssb-ai/models/best.pt)` matches
`runs/yolo26s_det_v2/weights/best.pt` exactly (`c90bd177…`), and the loaded model reports
`nc=5` with the V2 class names. *(This section previously claimed "the live model + Postgres are
still V1" with a pending cutover checklist — that was stale; the checklist had already been run.)*

---

## V1 — `nc=4` (2026-06-24) — **retired 2026-07-02 (superseded by V2)**

- **Dataset:** `gorengan-conter.yolo26` (Roboflow project `gorengan-conter`), `nc=4`:
  `['kosong', 'meja', 'penuh', 'sedikit']`. Splits: 163 train / 16 valid / 13 test. *(Dataset deleted 2026-07-02 when V2 replaced it; still recoverable from Roboflow.)*
- **Weights:** kept only as `backend-ssb-ai/models/best_v1_backup.pt` for rollback — no longer live.
- **Metrics:** mAP50 ≈ **0.949** on the test set. Not comparable to V2/V2.1: different class
  scheme (`nc=4`, had `kosong`) *and* the same random-split leakage issue described above.
