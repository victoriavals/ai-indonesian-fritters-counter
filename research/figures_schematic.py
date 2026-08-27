"""Gambar 1 dan Gambar 2 - diagram skematik untuk bab METODE PENELITIAN.

Template Simika MEWAJIBKAN kerangka penelitian digambarkan sebagai flowchart/blok diagram
(lihat Jurnal/00-PROFIL-JURNAL.md). Gambar 1 memenuhi syarat itu; Gambar 2 menjelaskan
normalisasi relatif-anchor yang menjadi inti pipeline pemetaan slot.

Keduanya digambar dengan matplotlib memakai Times New Roman (font jurnal) pada 300 dpi,
sehingga tampil konsisten dengan teks naskah dan tajam saat dicetak.

OUTPUT
------
- Jurnal/figures/gambar-1-alur-penelitian.png
- Jurnal/figures/gambar-2-anchor-normalisasi.png

JALANKAN
--------
    uv run python research/figures_schematic.py
    uv run python research/figures_schematic.py --only 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # wajib: tanpa ini backend interaktif tkagg dipakai dan gagal headless

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT.parent / "Jurnal" / "figures"

TNR = "Times New Roman"
DPI = 300

# Palet abu-abu: jurnal dicetak hitam-putih, jadi bedakan dengan nilai terang, bukan warna.
ISI = "#e8e8e8"
ISI_TEKAN = "#c4c4c4"
GARIS = "#222222"


def _siapkan() -> None:
    plt.rcParams.update({
        "font.family": TNR,
        "font.size": 9,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _kotak(ax, x, y, w, h, teks, isi=ISI, ukuran=9, tebal=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=isi, edgecolor=GARIS, linewidth=0.9,
    ))
    ax.text(x + w / 2, y + h / 2, teks, ha="center", va="center",
            fontsize=ukuran, fontfamily=TNR,
            fontweight="bold" if tebal else "normal", linespacing=1.35)


def _panah(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GARIS, linewidth=0.9,
                                shrinkA=1, shrinkB=1))


def gambar_1() -> Path:
    """Flowchart tahapan penelitian - diminta eksplisit oleh template Simika.

    Tata letak DUA KOLOM, bukan satu kolom memanjang. Alasannya praktis: lebar teks jurnal
    hanya 150 mm dan naskah dibatasi 8-12 halaman, sehingga flowchart satu kolom setinggi
    8 inci akan memakan hampir satu halaman penuh. Versi dua kolom memuat informasi yang
    sama dalam sekitar 0,35 halaman.
    """
    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)
    ax.axis("off")

    langkah = [
        ("1. Akuisisi Data CCTV@(EZVIZ, 768x432)", ISI),
        ("2. Anotasi 5 Kelas Ordinal@(661 citra, 12.200 instance)", ISI),
        ("3. Konversi Segmentasi@ke Kotak Pembatas", ISI),
        ("4. Deteksi Kebocoran@(MAE thumbnail 64x64)", ISI_TEKAN),
        ("5. Pemisahan Group-Aware@(union-find, 501/80/80)", ISI_TEKAN),
        ("6. Pelatihan Model@(seed & hyperparameter dikunci)", ISI),
        ("7. Evaluasi@(P, R, mAP50, mAP50-95)", ISI),
        ("8. Perbandingan Model@& Benchmark Komputasi", ISI),
    ]

    # Geometri dihitung agar baris terakhir TIDAK jatuh ke y negatif (yang membuat
    # kotak terpotong): y_baris3 = y_atas - tinggi - 3*(tinggi + jarak_v) harus > 0,1
    # untuk menyisakan ruang keterangan di bawah.
    lebar, tinggi = 0.42, 0.17
    jarak_v = 0.065
    x_kiri, x_kanan = 0.02, 0.56
    y_atas = 0.98

    posisi = []
    for i, (teks, isi) in enumerate(langkah):
        kolom, baris = i // 4, i % 4
        x = x_kiri if kolom == 0 else x_kanan
        y = y_atas - tinggi - baris * (tinggi + jarak_v)
        _kotak(ax, x, y, lebar, tinggi, teks.replace("@", chr(10)), isi=isi, ukuran=8)
        posisi.append((x, y))
        if baris < 3:
            _panah(ax, x + lebar / 2, y, x + lebar / 2, y - jarak_v)

    # Penghubung antar kolom. Panah diagonal langsung akan menembus kotak 4 dan membuat
    # teksnya tidak terbaca, jadi jalurnya dibuat ortogonal melalui gutter di antara kedua
    # kolom (x = 0,50, yang kosong karena kolom kiri berakhir di 0,44 dan kanan mulai 0,56).
    x4, y4 = posisi[3]
    x5, y5 = posisi[4]
    x_gutter = 0.50
    y_keluar = y4 + tinggi / 2
    # Masuk lewat SISI KIRI kotak 5, bukan sisi atas: jalur lewat atas akan melewati
    # y > 1,0 dan terpotong batas kanvas.
    y_masuk = y5 + tinggi / 2
    ax.plot([x4 + lebar, x_gutter], [y_keluar, y_keluar], color=GARIS, linewidth=0.9)
    ax.plot([x_gutter, x_gutter], [y_keluar, y_masuk], color=GARIS, linewidth=0.9)
    _panah(ax, x_gutter, y_masuk, x5, y_masuk)

    ax.text(0.5, -0.012,
            "Kotak berarsir gelap = kontribusi metodologis penelitian ini",
            ha="center", fontsize=7.4, fontfamily=TNR, style="italic")

    out = FIG_DIR / "gambar-1-alur-penelitian.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def gambar_2() -> Path:
    """Skema anchor 'meja' dan normalisasi koordinat relatif."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.set_xlim(-0.08, 1.30)
    ax.set_ylim(-0.22, 1.12)
    ax.axis("off")

    # Kotak anchor 'meja'
    ax.add_patch(mpatches.Rectangle((0, 0), 1.0, 0.78, facecolor="none",
                                    edgecolor=GARIS, linewidth=1.6))
    ax.text(0.5, 0.83, "kotak anchor 'meja'  (ax1, ay1) - (ax2, ay2)",
            ha="center", fontsize=9, fontfamily=TNR, fontweight="bold")

    # 20 slot 2x10
    for r in range(2):
        for c in range(10):
            cx = (c + 0.5) / 10
            # Koordinat citra berorigin KIRI-ATAS (ay1 = tepi atas), sedangkan sumbu-y
            # matplotlib berorigin di bawah. Baris 0 (Nampan 1-10) karena itu digambar
            # di BAGIAN ATAS - tanpa pembalikan ini gambar bertentangan dengan pemetaan
            # row-major yang justru ingin dijelaskannya.
            cy = 0.78 * (1 - (r + 0.5) / 2)
            ax.add_patch(mpatches.Circle((cx, cy), 0.028, facecolor=ISI,
                                         edgecolor=GARIS, linewidth=0.7))
            ax.text(cx, cy, str(r * 10 + c + 1), ha="center", va="center",
                    fontsize=5.4, fontfamily=TNR)

    # Satu deteksi contoh + proyeksi ke sumbu
    dx, dy = 0.35, 0.195           # sebuah nampan di baris bawah
    ax.plot([dx], [dy], marker="x", markersize=9, color="#b00000", markeredgewidth=1.8)
    # Proyeksi diukur dari tepi ATAS (ay1) dan tepi KIRI (ax1) - sesuai koordinat citra.
    ax.plot([dx, dx], [0.78, dy], linestyle=":", color="#b00000", linewidth=0.9)
    ax.plot([0, dx], [dy, dy], linestyle=":", color="#b00000", linewidth=0.9)
    ax.text(dx + 0.03, dy - 0.10, "centroid deteksi",
            fontsize=8, fontfamily=TNR, color="#b00000")
    ax.plot([0], [0.78], marker="o", markersize=4, color=GARIS)
    ax.text(0.012, 0.735, "$(ax_1, ay_1)$", fontsize=7.5, fontfamily=TNR, va="top")

    # Label dimensi
    ax.annotate("", xy=(1.0, -0.09), xytext=(0, -0.09),
                arrowprops=dict(arrowstyle="<|-|>", color=GARIS, linewidth=0.8))
    ax.text(0.5, -0.155, "$a_w = ax_2 - ax_1$", ha="center", fontsize=8.5, fontfamily=TNR)
    ax.annotate("", xy=(-0.045, 0.78), xytext=(-0.045, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=GARIS, linewidth=0.8))
    ax.text(-0.062, 0.39, "$a_h$", ha="center", va="center", rotation=90,
            fontsize=8.5, fontfamily=TNR)

    # Rumus normalisasi
    ax.text(1.06, 0.52,
            "Normalisasi relatif-anchor:\n\n"
            r"$r_x = \dfrac{c_x - ax_1}{a_w}$" "\n\n"
            r"$r_y = \dfrac{c_y - ay_1}{a_h}$",
            fontsize=9, fontfamily=TNR, va="center", ha="left")
    ax.text(1.06, 0.06,
            "Karena posisi disimpan relatif\n"
            "terhadap anchor, kalibrasi tetap\n"
            "sahih ketika meja bergeser di\n"
            "dalam bingkai kamera.",
            fontsize=7.6, fontfamily=TNR, va="center", ha="left", style="italic")

    out = FIG_DIR / "gambar-2-anchor-normalisasi.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hasilkan Gambar 1 dan 2 (skematik).")
    ap.add_argument("--only", type=int, choices=(1, 2), help="hasilkan satu gambar saja")
    args = ap.parse_args()

    _siapkan()
    tugas = {1: gambar_1, 2: gambar_2}
    pilih = [args.only] if args.only else sorted(tugas)

    for n in pilih:
        try:
            out = tugas[n]()
        except Exception as exc:  # noqa: BLE001
            print(f"[GAGAL] Gambar {n}: {exc}")
            sys.exit(1)
        kb = out.stat().st_size / 1024
        print(f"[OK] Gambar {n}: {out.name}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
