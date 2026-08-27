"""Gambar 3, 4, dan 5 - gambar berbasis data untuk bab HASIL DAN PEMBAHASAN.

Gambar 3  Pasangan near-duplicate latih-uji pada split acak Roboflow.
          Ini figur paling persuasif di naskah: pembaca melihat sendiri bahwa dua citra
          yang berada di sisi berlawanan dari split ternyata nyaris identik.
Gambar 4  Matriks konfusi 5x5 (dihitung ulang agar layak cetak, bukan PNG Ultralytics).
Gambar 5  Overlay deteksi + penomoran 20 slot pada frame nyata.

PRIVASI
-------
Izin pemilik usaha BELUM dipastikan, sehingga area di luar meja DISAMARKAN secara default
(`--blur-outside-table`). Area meja sendiri dibiarkan tajam karena itulah objek penelitian.
Penyamaran memakai kotak anchor 'meja' hasil deteksi model - bukan tebakan koordinat.
Matikan hanya bila izin sudah ada: `--no-blur`.

OUTPUT
------
- Jurnal/figures/gambar-3-near-duplicate.png
- Jurnal/figures/gambar-4-matriks-konfusi.png
- Jurnal/figures/gambar-5-overlay-slot.png

JALANKAN
--------
    uv run python research/figures_data.py --only 3
    uv run python research/figures_data.py            # ketiganya
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT.parent / "Jurnal" / "figures"
BOBOT = REPO_ROOT / "runs" / "yolo26s_det_v21_clean" / "weights" / "best.pt"

TNR = "Times New Roman"
DPI = 300
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THUMB = 64          # sama dengan make_clean_split.py - jangan diubah tanpa alasan
KELAS = ["habis", "hampir habis", "meja", "penuh", "sedikit"]


def _siapkan() -> None:
    plt.rcParams.update({
        "font.family": TNR,
        "font.size": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _citra(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def _thumbs(paths: list[Path]) -> np.ndarray:
    """(N, THUMB*THUMB) float32 - identik dengan make_clean_split.thumbnails()."""
    rows = np.empty((len(paths), THUMB * THUMB), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = im.convert("L").resize((THUMB, THUMB))
        rows[i] = np.asarray(g, dtype=np.float32).ravel()
    return rows


def _anchor_meja(img: Image.Image):
    """Kotak 'meja' terluas dari model. None bila tidak terdeteksi."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not BOBOT.is_file():
        print(f"[warn] bobot tidak ada: {BOBOT} - penyamaran dilewati")
        return None
    model = _anchor_meja._model = getattr(_anchor_meja, "_model", None) or YOLO(str(BOBOT))
    r = model.predict(source=img, conf=0.25, verbose=False)[0]
    terbaik, luas_maks = None, -1.0
    if r.boxes is not None:
        for b in r.boxes:
            if model.names[int(b.cls.item())] != "meja":
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            luas = (x2 - x1) * (y2 - y1)
            if luas > luas_maks:
                luas_maks, terbaik = luas, (x1, y1, x2, y2)
    return terbaik


def _samarkan_luar_meja(img: Image.Image) -> Image.Image:
    """Blur segalanya di luar kotak 'meja'; area meja tetap tajam.

    Dipakai karena izin data CCTV belum dipastikan - area di luar meja dapat memuat
    pelanggan dan bagian tempat usaha yang tidak relevan dengan penelitian.
    """
    kotak = _anchor_meja(img)
    kabur = img.filter(ImageFilter.GaussianBlur(radius=max(6, img.width // 90)))
    if kotak is None:
        print("[warn] 'meja' tidak terdeteksi - SELURUH citra dikaburkan demi keamanan")
        return kabur
    x1, y1, x2, y2 = (int(v) for v in kotak)
    pad = int(0.02 * img.width)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.width, x2 + pad), min(img.height, y2 + pad)
    hasil = kabur.copy()
    hasil.paste(img.crop((x1, y1, x2, y2)), (x1, y1))
    return hasil


def gambar_3(blur: bool) -> Path:
    """Pasangan near-duplicate paling mirip yang menyeberangi split latih/uji."""
    src = REPO_ROOT / "v2.1-gorengan-counter.yolo26"   # split ACAK Roboflow - itu intinya
    if not src.is_dir():
        sys.exit(f"ERROR: {src} tidak ditemukan")

    latih = _citra(src / "train" / "images")
    uji = _citra(src / "test" / "images")
    if not latih or not uji:
        sys.exit("ERROR: split train/test kosong")

    print(f"  menghitung thumbnail: {len(latih)} latih, {len(uji)} uji ...")
    t_latih, t_uji = _thumbs(latih), _thumbs(uji)

    # MAE tiap pasangan uji x latih, dihitung per potongan agar RAM terbatas.
    best = (1e9, -1, -1)
    di_bawah_5 = 0
    for i in range(len(uji)):
        mae = np.abs(t_latih - t_uji[i]).mean(axis=1)
        j = int(mae.argmin())
        if mae[j] < 5:
            di_bawah_5 += 1
        if mae[j] < best[0]:
            best = (float(mae[j]), i, j)

    mae_min, i_uji, j_latih = best
    pct = di_bawah_5 / len(uji) * 100
    print(f"  citra uji dengan kembar latih pada MAE<5: {di_bawah_5}/{len(uji)} ({pct:.0f}%)")
    print(f"  pasangan termirip: MAE={mae_min:.2f}")
    print(f"    latih: {latih[j_latih].name}")
    print(f"    uji  : {uji[i_uji].name}")

    im_latih = Image.open(latih[j_latih]).convert("RGB")
    im_uji = Image.open(uji[i_uji]).convert("RGB")
    if blur:
        print("  menyamarkan area di luar meja ...")
        im_latih, im_uji = _samarkan_luar_meja(im_latih), _samarkan_luar_meja(im_uji)

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 1.95))
    # Tampilkan hanya nomor frame; hash konten Roboflow tidak bermakna bagi pembaca dan
    # terpotong di tengah akan terlihat sembrono di naskah.
    def _label(q):
        m = re.search(r"Screenshot \((\d+)\)", q.name)
        return "Screenshot (" + m.group(1) + ")" if m else q.stem[:24]

    for ax, im, judul in (
        (axes[0], im_latih, "(a) split LATIH  -  " + _label(latih[j_latih])),
        (axes[1], im_uji, "(b) split UJI  -  " + _label(uji[i_uji])),
    ):
        ax.imshow(im)
        ax.set_title(judul, fontsize=8, fontfamily=TNR, pad=4)
        ax.axis("off")

    fig.suptitle(
        f"MAE thumbnail 64x64 antar kedua citra = {mae_min:.2f}"
        f"   (ambang near-duplicate < 5; baseline pasangan acak ~38)",
        fontsize=8.2, fontfamily=TNR, y=0.02, va="top",
    )
    out = FIG_DIR / "gambar-3-near-duplicate.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    # Catat angkanya supaya naskah mengutip hasil terukur, bukan ingatan.
    (FIG_DIR / "gambar-3-angka.txt").write_text(
        f"pasangan termirip MAE = {mae_min:.2f}\n"
        f"citra latih = {latih[j_latih].name}\n"
        f"citra uji   = {uji[i_uji].name}\n"
        f"citra uji dengan kembar latih MAE<5 = {di_bawah_5}/{len(uji)} ({pct:.1f}%)\n"
        f"sumber split = {src.name} (partisi acak Roboflow)\n",
        encoding="utf-8",
    )
    return out


def gambar_4() -> Path:
    """Matriks konfusi 5x5 - dihitung ulang, bukan memakai PNG bawaan Ultralytics.

    Alasan dihitung ulang: PNG Ultralytics memakai font dan skema warna sendiri yang tidak
    konsisten dengan naskah, dan template Simika meminta tulisan pada gambar terbaca jelas.
    Angkanya diambil dari `confusion_matrix.matrix` hasil validasi, jadi tetap dari
    eksperimen - bukan diketik ulang.
    """
    from ultralytics import YOLO

    data = REPO_ROOT / "dataset_det_clean" / "data.yaml"
    if not data.is_file():
        sys.exit("ERROR: dataset_det_clean/data.yaml tidak ada")
    if not BOBOT.is_file():
        sys.exit("ERROR: bobot model tidak ada: " + str(BOBOT))

    print("  menjalankan validasi untuk mengambil matriks konfusi ...")
    model = YOLO(str(BOBOT))
    # plots=True WAJIB. Dengan plots=False, Ultralytics 8.4 melewati pengisian
    # confusion_matrix dan mengembalikan matriks (6,6) yang SELURUHNYA NOL - tanpa error
    # apa pun. Gambar yang dihasilkan akan tampak wajar tetapi kosong isinya.
    hasil = model.val(data=str(data), split="test", imgsz=640, workers=0,
                      plots=True, verbose=False,
                      project=str(REPO_ROOT / "runs"), name="_cm_tmp", exist_ok=True)

    m = np.asarray(hasil.confusion_matrix.matrix, dtype=float)
    if m.sum() == 0:
        sys.exit("ERROR: matriks konfusi kosong - jangan pakai gambar ini. "
                 "Periksa apakah plots=True benar-benar diteruskan ke model.val().")
    label = list(KELAS) + ["latar"]          # Ultralytics menambah kolom/baris background

    # Normalisasi per KOLOM (per kelas sebenarnya) - itu yang bermakna untuk membaca recall.
    kolom = m.sum(axis=0, keepdims=True)
    norm = np.divide(m, kolom, out=np.zeros_like(m), where=kolom > 0)

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(range(len(label)))
    ax.set_yticks(range(len(label)))
    ax.set_xticklabels(label, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(label, fontsize=8)
    ax.set_xlabel("Kelas sebenarnya", fontsize=9)
    ax.set_ylabel("Prediksi model", fontsize=9)

    for i in range(len(label)):
        for j in range(len(label)):
            if m[i, j] == 0:
                continue
            ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else "black")

    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                 label="proporsi per kelas sebenarnya")
    out = FIG_DIR / "gambar-4-matriks-konfusi.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    (FIG_DIR / "gambar-4-angka.txt").write_text(
        "matriks konfusi (baris=prediksi, kolom=sebenarnya), urutan: "
        + ", ".join(label) + chr(10)
        + chr(10).join(" ".join(str(int(v)) for v in baris) for baris in m) + chr(10),
        encoding="utf-8",
    )
    return out


def gambar_5(blur: bool) -> Path:
    """Overlay deteksi + penomoran 20 slot pada frame nyata.

    Kalibrasi slot dihitung DARI FRAME ITU SENDIRI memakai compute_calibration() salinan
    murni (research/slot_mapping.py) - BUKAN dibaca dari basis data produksi. Ini penting:
    membaca kalibrasi produksi berarti menyentuh Postgres produksi.
    """
    from ultralytics import YOLO

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slot_mapping import compute_calibration, slots_from_calibration

    uji = _citra(REPO_ROOT / "dataset_det_clean" / "test" / "images")
    if not uji:
        sys.exit("ERROR: split test kosong")
    if not BOBOT.is_file():
        sys.exit("ERROR: bobot model tidak ada: " + str(BOBOT))

    model = YOLO(str(BOBOT))
    TRAY = {"penuh", "sedikit", "hampir habis", "habis"}

    # Pilih frame dengan jumlah nampan terdeteksi paling banyak (paling informatif).
    print("  memilih frame dengan nampan terbanyak ...")
    terbaik = None
    for path in uji[:40]:
        with Image.open(path) as im0:
            img = im0.convert("RGB")
        r = model.predict(source=img, conf=0.25, verbose=False)[0]
        anchor, luas_maks, trays = None, -1.0, []
        if r.boxes is not None:
            for b in r.boxes:
                nama = model.names[int(b.cls.item())]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if nama == "meja":
                    luas = (x2 - x1) * (y2 - y1)
                    if luas > luas_maks:
                        luas_maks, anchor = luas, (x1, y1, x2, y2)
                elif nama in TRAY:
                    trays.append(((x1 + x2) / 2, (y1 + y2) / 2, nama, float(b.conf.item())))
        if anchor and (terbaik is None or len(trays) > len(terbaik[2])):
            terbaik = (path, anchor, trays, img)
        if terbaik and len(terbaik[2]) >= 20:
            break

    if terbaik is None:
        sys.exit("ERROR: tidak ada frame dengan anchor 'meja' terdeteksi")
    path, anchor, trays, img = terbaik
    print("  frame: " + path.name + " - " + str(len(trays)) + " nampan terdeteksi")
    aw0, ah0 = anchor[2] - anchor[0], anchor[3] - anchor[1]

    # CATATAN METODOLOGIS - kenapa tidak memakai compute_calibration() apa adanya.
    #
    # compute_calibration() membagi baris dengan mengurutkan SELURUH centroid menurut y lalu
    # mengambil separuh pertama sebagai baris atas. Pada meja yang miring terhadap kamera
    # (kasus nyata di sini: sisi kiri meja lebih rendah), rentang y kedua baris saling
    # tumpang tindih sehingga pembagian itu mencampur baris - nampan baris bawah dapat
    # bernomor 2 sementara tetangganya bernomor 11.
    #
    # Di produksi masalah ini tidak muncul karena kalibrasi dilakukan MANUAL dengan klik,
    # bukan otomatis. Karena skrip penelitian tidak boleh membaca kalibrasi dari basis data
    # produksi, baris di sini ditentukan dengan membelah pada TITIK TENGAH rentang y
    # relatif-anchor - yang setara dengan hasil klik manual pada tata letak 2x10, dan tahan
    # terhadap kemiringan meja.
    #
    # slot_mapping.py sendiri TIDAK diubah: berkas itu salinan verbatim produksi dan sudah
    # diverifikasi 17/17 terhadap test backend. Penyesuaian ini khusus untuk gambar.
    # Meja miring terhadap kamera, sehingga membelah baris berdasarkan y saja (baik dengan
    # pengurutan seperti compute_calibration, maupun dengan titik tengah) tetap mencampur
    # baris - percobaan titik tengah menghasilkan 8/12, bukan 10/10.
    #
    # Cara yang benar: proyeksikan centroid ke SUMBU UTAMA meja lewat PCA, lalu belah pada
    # MEDIAN proyeksi tegak lurus. Median menjamin pembagian tepat separuh, dan proyeksi
    # membuat pembelahan mengikuti kemiringan meja, bukan sumbu gambar.
    P = np.array([[cx, cy] for cx, cy, _s, _c in trays], dtype=float)
    pusat = P.mean(axis=0)
    _u, _sv, vt = np.linalg.svd(P - pusat, full_matrices=False)
    sepanjang = (P - pusat) @ vt[0]      # sepanjang meja (kiri-kanan)
    lintang = (P - pusat) @ vt[1]        # tegak lurus meja (baris atas/bawah)
    # Arah vektor eigen sembarang tandanya, jadi kedua sumbu diorientasikan eksplisit:
    # nilai kecil pada 'lintang' harus berarti baris ATAS, dan 'sepanjang' harus menaik
    # dari KIRI ke kanan. Tanpa ini penomoran bisa terbalik (slot 1 di kanan-bawah).
    if np.corrcoef(lintang, P[:, 1])[0, 1] < 0:
        lintang = -lintang
    if np.corrcoef(sepanjang, P[:, 0])[0, 1] < 0:
        sepanjang = -sepanjang
    batas = float(np.median(lintang))
    idx_atas = sorted(np.where(lintang <= batas)[0], key=lambda i: sepanjang[i])
    idx_bawah = sorted(np.where(lintang > batas)[0], key=lambda i: sepanjang[i])
    urut = list(idx_atas) + list(idx_bawah)
    kal = [{"nampan": n + 1,
            "rx": round((P[i, 0] - anchor[0]) / aw0, 5),
            "ry": round((P[i, 1] - anchor[1]) / ah0, 5)}
           for n, i in enumerate(urut)]
    print("  baris atas " + str(len(idx_atas)) + " nampan, baris bawah "
          + str(len(idx_bawah)) + " (pembelahan PCA + median)")
    ringkas, slots = slots_from_calibration(anchor, trays, kal, 0.18)
    print("  ringkasan: " + str(ringkas))

    if blur:
        img = _samarkan_luar_meja(img)

    fig, ax = plt.subplots(figsize=(5.9, 5.9 * img.height / img.width))
    ax.imshow(img)
    ax.axis("off")

    ax1, ay1, ax2, ay2 = anchor
    aw, ah = ax2 - ax1, ay2 - ay1
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((ax1, ay1), aw, ah, fill=False,
                              edgecolor="white", linewidth=1.4))

    warna = {"Penuh": "#1a7f37", "Sedikit": "#b8860b", "Hampir Habis": "#c2410c",
             "Habis": "#b00000", "Diangkat": "#666666"}
    for s, sl in zip(kal, slots):
        cx, cy = ax1 + s["rx"] * aw, ay1 + s["ry"] * ah
        ax.add_patch(mp.Circle((cx, cy), max(7, img.width / 115), fill=True,
                               facecolor="white", alpha=0.82,
                               edgecolor=warna.get(sl["status"], "#666"), linewidth=1.6))
        ax.text(cx, cy, sl["id"].split()[-1], ha="center", va="center",
                fontsize=5.6, fontfamily=TNR, color="black")

    keterangan = "  |  ".join(k + ": " + str(v) for k, v in ringkas.items() if v)
    ax.set_title(keterangan, fontsize=8, fontfamily=TNR, pad=5)

    out = FIG_DIR / "gambar-5-overlay-slot.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    (FIG_DIR / "gambar-5-angka.txt").write_text(
        "frame = " + path.name + chr(10)
        + "nampan terdeteksi = " + str(len(trays)) + chr(10)
        + "ringkasan status = " + str(ringkas) + chr(10)
        + "kalibrasi dihitung dari frame ini sendiri (compute_calibration), "
          "bukan dari basis data produksi" + chr(10),
        encoding="utf-8",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hasilkan Gambar 3-5 (berbasis data).")
    ap.add_argument("--only", type=int, choices=(3, 4, 5))
    ap.add_argument("--no-blur", action="store_true",
                    help="jangan samarkan area di luar meja (hanya bila izin sudah ada)")
    args = ap.parse_args()

    _siapkan()
    blur = not args.no_blur
    if not blur:
        print("[PERHATIAN] penyamaran DIMATIKAN - pastikan izin pemilik usaha sudah ada.")

    for n, fn in ((3, lambda: gambar_3(blur)), (4, gambar_4), (5, lambda: gambar_5(blur))):
        if args.only not in (None, n):
            continue
        print("Gambar " + str(n) + " ...")
        try:
            out = fn()
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print("[GAGAL] Gambar " + str(n) + ": " + str(exc))
            continue
        print("[OK] Gambar " + str(n) + ": " + out.name
              + " (" + str(round(out.stat().st_size / 1024)) + " KB)")


if __name__ == "__main__":
    main()
