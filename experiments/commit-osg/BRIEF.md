# Brief Implementasi: YOLO26-OSG (Ordinal State Grading)

Kamu adalah Claude Code yang menjalankan proyek riset ini dari nol sampai selesai di mesin Windows dengan GPU RTX 4060 Ti 8GB. Dokumen ini adalah satu-satunya sumber kebenaran; kamu tidak punya akses ke percakapan sebelumnya.

---

## ATURAN KERJA (baca dulu, patuhi selalu)

1. **Satu STEP per giliran.** Kerjakan satu STEP sampai tuntas, tulis laporannya, lalu **BERHENTI** dan tunggu user mengetik `next`. Jangan pernah menyambung ke STEP berikutnya sendiri.
2. **Jangan pernah mengarang angka.** Tidak ada nilai placeholder, tidak ada contoh, tidak ada "kira-kira". Kalau sebuah angka belum dihitung, tulis `NOT_YET_COMPUTED`.
3. **Gerbang itu mengikat.** STEP 2, 3, dan 6 punya kriteria lulus. Kalau gagal, **berhenti dan laporkan**. Jangan akali, jangan longgarkan ambangnya, jangan lanjut diam-diam.
4. **Hasil negatif dilaporkan apa adanya.** Kalau sebuah modul tidak menolong, atau lengan pembanding menang, itu temuan yang sah. Jangan disembunyikan, jangan dipoles.
5. **Simpan inkremental.** Setiap STEP menulis hasilnya ke disk sebelum selesai. Crash tidak boleh menghilangkan pekerjaan sebelumnya.
6. **Catat semuanya** ke `RESULTS/RUNLOG.md`: perintah yang dijalankan, waktu mulai dan selesai, versi paket, seed, dan setiap kejadian tak terduga.
7. Kalau ada yang ambigu di brief ini, **tanya user**, jangan menebak.

---

## KONTEKS RISET

### Masalah
Meja buffet gorengan dipantau satu kamera CCTV. Ada 20 nampan yang tingkat isinya harus diketahui terus-menerus. Tingkat isi punya empat tingkat **ordinal** berbasis persentase volume.

### Kenapa arsitekturnya diubah
Dua asumsi detektor generik patah pada masalah ini:

1. **Kelas dianggap berkorelasi dengan geometri.** Di sini nampan `habis` dan `penuh` punya bounding box **identik**. Geometri membawa nol informasi kelas. Tapi cabang klasifikasi dan regresi YOLO berbagi fitur neck yang sama, dan fitur itu dioptimalkan box loss agar peka tepi.
2. **Instance dianggap independen.** Padahal 20 nampan dalam satu bingkai berbagi pencahayaan, exposure, dan jam. Penilaian absolut jadi tidak stabil; penilaian relatif lebih stabil.

Bukti dari penelitian sebelumnya pada dataset yang sama (naskah SIMIKA, under review):
- 100% kesalahan klasifikasi meleset **persis satu tingkat**. Tidak ada yang melompat dua tingkat.
- Kelas `meja` (murni geometris) mencapai mAP50-95 **0,993**, sementara `sedikit` hanya **0,834**.
- Simpangan baku antar-seed pada mAP50: **0,0090**.

### Arsitektur usulan: deteksi sekali, nilai secara sparse

**Tahap A (deteksi agnostik-kelas).** YOLO26s standar, tetapi hanya membedakan **dua kelas geometris**: `nampan` dan `meja`. Tidak tahu tingkat isi sama sekali.

**Tahap B (penilai ordinal sparse).** Berjalan hanya pada K=32 instance berskor tertinggi per bingkai:
- **B1 Tekstur**: RoI-Align pada fitur stride-4, lalu compact bilinear pooling (statistik orde dua)
- **B2 Kalibrasi**: self-attention lintas 32 deskriptor, dengan positional encoding dari posisi relatif terhadap anchor meja
- **B3 Head ordinal**: tiga logit kumulatif CORAL dengan bias terurut

---

## KEPUTUSAN DESAIN IMPLEMENTASI (penting, harus didokumentasikan)

**Tahap A dibekukan (frozen) saat melatih tahap B.**

Konsekuensinya harus dipahami dan dicatat:
- **Keuntungan:** metrik tahap A dijamin identik persis di seluruh konfigurasi R0b sampai R5. Ini menjadikannya kontrol integritas eksperimen yang sempurna. Kalau metrik tahap A pernah berbeda antar konfigurasi, ada bug.
- **Keuntungan kedua:** tahap A hanya dilatih sekali per fold, lalu fitur RoI di-cache. Semua konfigurasi tahap B dilatih di atas cache itu, hitungan menit bukan jam. Anggaran total turun dari sekitar 70 jam menjadi sekitar 20 jam.
- **Konsekuensi ilmiah:** ini **bukan** pelatihan end-to-end. Harus dinyatakan eksplisit di bagian Methods naskah, bukan didiamkan. Tulis: "Stage A is trained once per fold and kept frozen during Stage B training."

Kalau kamu menemukan alasan teknis kuat untuk tidak membekukan, **laporkan ke user dan tunggu keputusan**, jangan ubah sendiri.

---

## LOKASI DAN STRUKTUR

Pekerjaan ini masuk ke repo yang sudah ada: **`ai-indonesian-fritters-counter`**, yang sudah berisi kode training era naskah SIMIKA.

**Aturan yang mengikat:** jangan ubah, timpa, atau refactor apa pun di luar `experiments/commit-osg/`. Skrip era SIMIKA harus tetap bisa direproduksi apa adanya, karena naskah itu masih dalam review dan reviewer bisa memintanya.

```
ai-indonesian-fritters-counter/
  scripts/              <- MILIK SIMIKA, JANGAN DIUBAH (boleh dipakai ulang)
  config.yaml           <- MILIK SIMIKA, JANGAN DIUBAH
  research/             <- MILIK SIMIKA, JANGAN DIUBAH
  CLAUDE.md             <- aturan kerja, dibaca otomatis tiap sesi
  experiments/
    commit-osg/         <- SELURUH pekerjaan ini di sini
      BRIEF.md          <- berkas ini
      config_osg.yaml
      data/
        stageA/         # label YOLO 2 kelas, per fold
        ordinal/        # label ordinal per instance (JSON)
        folds/          # definisi 5 fold + verifikasi kebocoran
      src/
        data_prep.py    # WRAPPER tipis di atas skrip konversi yang sudah ada
        fold_split.py
        stage_a.py
        features.py     # hook + RoI-Align + cache
        stage_b.py      # B1, B2, B3, varian
        metrics.py      # QWK, MAE-ordinal, matching, bootstrap
        run.py          # orkestrator per konfigurasi
        make_tables.py
        make_figures.py
      cache/            # fitur RoI, besar, WAJIB masuk .gitignore
      RESULTS/
        results.json
        tables/
        figures/
        RUNLOG.md
        ENV.txt
```

**Git.** Buat branch `commit-osg` di awal. Commit setiap akhir STEP dengan pesan `STEP n: <ringkasan>`. Jangan commit `cache/`, bobot model, atau citra dataset.

---

# STEP 0. Lingkungan dan verifikasi awal

**Kerjakan:**
1. **Survei repo yang sudah ada dulu.** Baca `README.md`, `TRAINING.md`, `MODELS.md`, `config.yaml`, dan isi `scripts/`. Laporkan:
   - Skrip konversi poligon ke bbox yang sudah ada: nama berkas, cara pakai, format masukan dan keluarannya
   - Bagaimana dependensi dikelola (repo ini memakai `uv`)
   - Konfigurasi training yang dipakai naskah SIMIKA
   Ini penting: STEP 1 akan **memakai ulang** skrip konversi itu, bukan menulis yang baru, supaya dua paper dengan dataset sama tidak memakai dua kode konversi berbeda.
2. Buat branch `commit-osg` dan folder `experiments/commit-osg/`.
3. Pakai lingkungan `uv` yang sudah ada di repo. Tambahkan hanya paket yang belum ada: `torchvision`, `scipy`, `scikit-learn`, `matplotlib`. Jangan ganti versi paket yang sudah terpasang, karena itu bisa merusak reproduktibilitas era SIMIKA. Kalau versi bentrok, **laporkan dan tunggu keputusan user**.
4. Verifikasi `torch.cuda.is_available()` True, cetak nama GPU dan VRAM.
5. Verifikasi versi `ultralytics` **mendukung YOLO26**. Kalau tidak, laporkan ke user dan berhenti.
6. Cek ruang disk kosong minimal 50 GB (cache fitur besar).
7. Tambahkan `experiments/commit-osg/cache/` dan `experiments/commit-osg/RESULTS/figures/*.png` ke `.gitignore` bila perlu.
8. Tulis `RESULTS/ENV.txt` berisi semua versi paket, versi CUDA, info GPU, versi Windows, dan commit hash repo saat ini.

**Catatan Windows yang wajib diperhatikan:**
- Ultralytics dataloader sering menggantung di Windows dengan `workers > 0`. Set `workers=0` atau maksimal 2, dan catat pilihannya.
- Bungkus semua entry point training dengan `if __name__ == "__main__":` karena Windows memakai spawn, bukan fork.
- Pakai path absolut. Hindari spasi di path proyek.

**Laporkan:** ringkasan lingkungan, versi ultralytics, apakah YOLO26 tersedia, ruang disk.

**BERHENTI. Tunggu `next`.**

---

# STEP 1. Intake data dan konversi label

**Input dari user.** Minta user menaruh ekspor dataset di `data/raw/`. Formatnya: ekspor segmentasi YOLO (poligon) dengan 5 kelas.

Urutan kelas asli:
```
0 = habis          (0%)
1 = hampir habis   (1-25%)
2 = sedikit        (26-50%)
3 = penuh          (51-100%)
4 = meja           (anchor spasial, tepat 1 per citra)
```

**Kerjakan:**

1. **Konversi poligon ke bounding box: PAKAI ULANG skrip yang sudah ada di `scripts/`**, hasil survei STEP 0. Tulis wrapper tipis di `src/data_prep.py` yang memanggilnya, jangan tulis ulang logikanya.

   Alasannya bukan kepraktisan. Naskah SIMIKA memakai skrip itu untuk dataset yang sama. Kalau kamu tulis ulang dan ada beda sekecil apa pun, dua paper akan mendeskripsikan konversi yang "sama" dari dua kode berbeda, dan itu celah yang tidak enak dijelaskan ke reviewer.

   Rumus yang dipakai skrip itu seharusnya:
   ```
   cx = (min_x + max_x)/2,  cy = (min_y + max_y)/2
   w  = max_x - min_x,      h  = max_y - min_y
   ```
   **Verifikasi bahwa skrip yang ada memang memakai rumus ini.** Kalau berbeda, laporkan bedanya ke user dan tunggu keputusan, jangan diam-diam pakai versimu sendiri.

   Kotak dengan `w < 0.001` atau `h < 0.001` dibuang; **catat berapa yang dibuang** (referensi: seharusnya nol).

2. **Buat dua tampilan label dari anotasi yang sama.**
   - `data/stageA/` : label YOLO **2 kelas**. Kelas 0..3 digabung jadi `nampan` (id 0), kelas 4 jadi `meja` (id 1).
   - `data/ordinal/labels.json` : untuk setiap instance `nampan`, simpan `{image_id, bbox_xyxy_norm, ordinal_level}` dengan `ordinal_level` = kelas asli 0..3. Instance `meja` **tidak** punya level ordinal.

3. **Verifikasi wajib.** Cetak dan bandingkan dengan angka rujukan berikut. Kalau meleset, **berhenti dan laporkan**, jangan lanjut.

   | Butir | Nilai rujukan |
   |---|---|
   | Jumlah citra | 661 |
   | Total instance | 12.200 |
   | habis | 269 |
   | hampir habis | 3.367 |
   | sedikit | 2.768 |
   | penuh | 5.135 |
   | meja | 661 |

4. **Verifikasi tambahan:** setiap citra harus punya **tepat satu** instance `meja`. Laporkan citra mana pun yang melanggar.

5. Ekstrak timestamp tiap citra (dari nama berkas atau EXIF), simpan ke `data/timestamps.json`. Ini dipakai untuk stratifikasi jam di STEP 7. Kalau tidak tersedia, **laporkan ke user** karena uji mekanisme B2 bergantung padanya.

**Laporkan:** tabel jumlah instance per kelas hasil hitung sendiri berdampingan dengan nilai rujukan, jumlah kotak yang dibuang, pelanggaran aturan satu-meja, ketersediaan timestamp.

**BERHENTI. Tunggu `next`.**

---

# STEP 2. Pembagian 5-fold group-aware. GERBANG 1

Citra berasal dari rekaman video berkelanjutan, jadi bingkai berdekatan nyaris kembar. Pembagian acak akan membocorkan data latih ke data uji.

**Kerjakan:**

1. Untuk setiap citra, buat thumbnail grayscale 64x64. Hitung jarak MAE antar semua pasangan:
   ```
   MAE(Ia, Ib) = (1/N) * sum |ta(k) - tb(k)|,  N = 4096
   ```
2. Hubungkan dua citra bila `MAE < 12`. Bentuk komponen terhubung dengan union-find. **Nilai rujukan: sekitar 182 klaster adegan.** Laporkan jumlah yang kamu dapat.
3. Bagi ke 5 fold **pada tingkat klaster**, bukan tingkat citra. Klaster terbesar ditempatkan lebih dulu, ke fold yang paling jauh di bawah kuotanya. Target: tiap fold sekitar 132 citra.
4. **Verifikasi kebocoran per fold.** Untuk setiap fold, hitung jarak MAE minimum dari tiap citra uji ke citra latih terdekat.
5. Simpan `data/folds/fold_{0..4}.json` dan `data/folds/leakage_verification.json`.

**KRITERIA LULUS GERBANG 1:**
- Setiap fold: **MAE minimum ≥ 12**
- Setiap fold: **nol** citra uji dengan MAE < 5
- Distribusi kelas tiap fold wajar (tidak ada fold tanpa instance `habis`)

Kalau gagal: **berhenti, laporkan fold mana dan angkanya.** Jangan turunkan ambang.

**Laporkan:** jumlah klaster, jumlah citra per fold, tabel verifikasi kebocoran (MAE minimum dan median per fold), distribusi kelas per fold.

**BERHENTI. Tunggu `next`.**

---

# STEP 3. Smoke test 10 epoch. GERBANG 2

Ini murah dan wajib. Tujuannya menemukan bug integrasi sebelum membakar puluhan jam.

**Kerjakan:**

1. **Latih tahap A, 10 epoch, fold 0 saja.** Pastikan pipeline jalan.

2. **Temukan lapisan fitur stride-4 secara empiris.** Jangan hardcode indeks lapisan. Jalankan satu forward pass dengan input 640x640, pasang hook di setiap lapisan backbone, cetak bentuk keluarannya, lalu identifikasi lapisan yang menghasilkan stride 4 (160x160 untuk input 640). Catat indeksnya ke `RUNLOG.md`.

3. **Verifikasi RoI-Align.** Ambil satu citra, jalankan tahap A, ambil top-K kotak, jalankan `torchvision.ops.roi_align` pada peta fitur stride-4. **Simpan gambar visualisasi**: citra asli dengan kotak digambar, berdampingan dengan crop RoI yang sesuai. Periksa mata bahwa crop benar-benar berisi nampan yang dimaksud, bukan bagian lain. Simpan ke `RESULTS/figures/roi_sanity_check.png`.

4. **Verifikasi monotonisitas CORAL.** Implementasikan head:
   ```python
   class CoralHead(nn.Module):
       def __init__(self, in_dim, num_levels=4):
           super().__init__()
           self.w = nn.Linear(in_dim, 1, bias=False)      # vektor bobot bersama
           self.b1 = nn.Parameter(torch.zeros(1))
           self.gaps = nn.Parameter(torch.zeros(num_levels - 2))
       def biases(self):
           g = F.softplus(self.gaps)                       # selalu positif
           return torch.cat([self.b1, self.b1 - torch.cumsum(g, 0)])
       def forward(self, f):
           return self.w(f) + self.biases()                # (N, K-1) logit kumulatif
   ```
   Loss: `BCEWithLogitsLoss` terhadap target `1[y > k]` untuk k = 1,2,3.
   Prediksi: `level = (torch.sigmoid(logits) > 0.5).sum(dim=1)`.

   **Uji:** dengan bobot acak, pada 1000 masukan acak, pastikan `P(y>1) >= P(y>2) >= P(y>3)` untuk **setiap** baris. Kalau ada satu saja yang melanggar, ada bug di parameterisasi bias.

5. **Verifikasi metrik.** Uji fungsi QWK, MAE-ordinal, dan 1-off accuracy dengan kasus buatan yang jawabannya sudah diketahui (misal: prediksi sempurna → QWK = 1,0; prediksi acak → QWK mendekati 0).

**KRITERIA LULUS GERBANG 2:**
- Tahap A latih tanpa error
- Lapisan stride-4 teridentifikasi
- Visualisasi RoI terlihat benar (user yang memutuskan)
- Monotonisitas CORAL: 1000/1000 lolos
- Fungsi metrik lolos uji kasus buatan

**Laporkan:** indeks lapisan stride-4 dan bentuk tensornya, hasil uji monotonisitas, hasil uji metrik, dan **tampilkan `roi_sanity_check.png` ke user untuk diperiksa**.

**BERHENTI. Tunggu `next`.**

---

# STEP 4. R0: baseline 5 kelas, penetap satuan ukur

R0 bukan pembanding. Fungsinya menghasilkan σ (simpangan baku) yang jadi satuan keputusan seluruh eksperimen, dan nilai ordinal baseline yang belum pernah dihitung siapa pun.

**Konfigurasi pelatihan** (sama untuk semua run tahap A, jangan diubah antar konfigurasi):
```
model      : yolo26s (pretrained)
imgsz      : 640
epochs     : 200, patience 50 (early stopping)
batch      : 16
optimizer  : auto
lr schedule: cosine, lrf 0.01
close_mosaic: 10
augment    : HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0, erasing 0.4
seed       : 0
workers    : 0 (Windows)
device     : 0
```

**Kerjakan:**
1. Latih YOLO26s **5 kelas asli** pada kelima fold. Seed 0.
2. Ulangi fold 0 dengan seed 1 dan seed 2 (untuk σ antar-seed).
3. Untuk setiap fold, hitung: precision, recall, mAP50, mAP50-95 (per kelas dan keseluruhan), **plus metrik ordinal** (QWK, MAE-ordinal, 1-off accuracy, proporsi lompat ≥2 tingkat) dengan protokol pencocokan Hungarian IoU ≥ 0,5.
4. Hitung σ antar-fold dan σ antar-seed untuk **setiap** metrik.
5. Simpan ke `RESULTS/results.json` di bawah kunci `R0`.

**Perkiraan waktu:** 8 run × sekitar 1,75 jam = sekitar 14 jam. Jalankan berurutan, simpan setelah tiap run.

**Laporkan:** tabel mean ± std per metrik, dan **secara eksplisit nilai 3σ untuk QWK**, karena itu ambang gerbang di STEP 6.

**BERHENTI. Tunggu `next`.**

---

# STEP 5. Tahap A (2 kelas) dan cache fitur

**Kerjakan:**
1. Latih tahap A **2 kelas** (`nampan`, `meja`) pada kelima fold, konfigurasi identik dengan STEP 4, seed 0.
2. Bekukan bobotnya. Simpan ke `models/stageA_fold{0..4}.pt`.
3. Untuk setiap citra di setiap fold (latih dan uji), jalankan tahap A, lalu:
   - Ambil top-K=32 kotak berdasarkan skor objectness
   - Cocokkan ke ground truth dengan IoU ≥ 0,5 (Hungarian) untuk mendapat `ordinal_level`
   - **Tambahkan kotak ground truth ke himpunan instance latih** (pola Fast R-CNN). Ini menangani beda distribusi kotak antara latih dan inferensi. Kotak GT **tidak** ditambahkan ke himpunan uji.
   - Ekstrak fitur stride-4, jalankan RoI-Align (output 7x7), simpan tensor
4. Cache ke `cache/fold{n}_{train,test}.pt`. Sertakan: fitur RoI, kotak, `ordinal_level`, `image_id`, koordinat relatif-anchor `(rx, ry)` terhadap kotak `meja` terlebar, dan timestamp.
5. Hitung metrik tahap A per fold dan simpan ke `results.json` sebagai `stageA_frozen`. **Angka ini harus muncul identik di seluruh konfigurasi berikutnya.**

**Perkiraan waktu:** 5 × 1,75 jam training + sekitar 20 menit caching = sekitar 9 jam.

**Laporkan:** metrik tahap A per fold, ukuran cache, jumlah instance ter-cache per fold, berapa persen instance uji yang berhasil ter-match ke GT.

**BERHENTI. Tunggu `next`.**

---

# STEP 6. R0b dan R1. GERBANG 3, gerbang penentu

Mulai sini semua pelatihan berjalan di atas cache, jadi cepat (menit, bukan jam).

**R0b: struktur dua tahap, head nominal.**
Tahap B minimal: RoI feature → global average pooling → MLP → **softmax 4 arah**. Tanpa B1, tanpa B2. Ini mengisolasi efek restrukturisasi dari efek ordinalitas.

**R1: R0b dengan head diganti CORAL.**
Persis sama dengan R0b, **satu-satunya perbedaan** adalah head nominal diganti head ordinal CORAL. Bukan yang lain.

**Kerjakan:**
1. Latih R0b pada kelima fold. Catat metrik ordinal.
2. Latih R1 pada kelima fold. Catat metrik ordinal.
3. Hitung ΔQWK antara R0b dan R1, dan nyatakan **dalam satuan σ** dari STEP 4.
4. Hitung juga ΔmAP tahap A. **Nilainya harus tepat nol** karena tahap A beku.

**KRITERIA GERBANG 3:**

| Hasil | Arti | Tindakan |
|---|---|---|
| ΔQWK > 3σ | Terkonfirmasi | Lanjut ke STEP 7 |
| 1σ < ΔQWK ≤ 3σ | Tidak konklusif | **Berhenti, laporkan, tunggu keputusan user** |
| ΔQWK ≤ 1σ | Tertolak | **Berhenti.** Premis R2 dan R3 gugur. Laporkan bahwa penyebabnya kemungkinan noise label di batas 25% dan 50%, bukan salah spesifikasi objektif |

**SINYAL BUG yang harus dikenali, bukan dirayakan:**
- Kalau mAP tahap A **berubah sama sekali** antara R0b dan R1: ada kebocoran antar tahap. Tahap A beku, mustahil berubah. **Berhenti dan debug.**
- Kalau QWK naik tajam **dan** mAP juga naik tajam: tidak masuk akal secara teori karena head ordinal tidak menyentuh lokalisasi. **Berhenti dan debug.**

**Laporkan:** tabel R0b lawan R1 untuk semua metrik ordinal, ΔQWK dalam satuan σ, verifikasi mAP tahap A identik, dan **keputusan gerbang secara eksplisit**.

**BERHENTI. Tunggu `next`.**

---

# STEP 7. R2 dan R3. Uji mekanisme

Jalankan **hanya** kalau Gerbang 3 lolos.

**R2 = R1 + B2 (kalibrasi relasional).**
Self-attention 1 lapis, 4 head, d=64, residual, lintas 32 deskriptor dalam satu bingkai. Positional encoding dari `(rx, ry)`.

**R3 = R2 + B1 (pooling orde dua).**
Ganti global average pooling dengan compact bilinear pooling pada fitur RoI.

**Kerjakan:**
1. Latih R2 pada kelima fold.
2. **Uji mekanisme B2:** stratifikasi citra uji ke 4 kuartil berdasarkan jam perekaman (rentang 02.57 sampai 20.06). Kuartil ekstrem = paling gelap dan paling terang. Hitung ΔQWK (R1 → R2) **per kuartil**. Hitung rasio perbaikan ekstrem terhadap normal.
3. Latih R3 pada kelima fold.
4. **Uji mekanisme B1:** hitung tingkat kesalahan **per pasangan kelas bertetangga** (habis↔hampir habis di batas 0/1%, hampir habis↔sedikit di batas 25%, sedikit↔penuh di batas 50%). Bandingkan R2 dan R3.
5. Verifikasi lagi metrik tahap A tetap identik.

**Yang harus dilaporkan apa adanya:**
- Kalau rasio kuartil ekstrem terhadap normal **≤ 1**, mekanisme B2 tidak terkonfirmasi. Perbaikannya kapasitas, bukan kalibrasi. Laporkan begitu.
- Kalau ΔB1 ≤ 1σ, laporkan bahwa B1 tidak membayar biayanya, sertakan parameter dan GFLOPs tambahannya.

**Laporkan:** tabel per kuartil untuk B2, tabel per pasangan kelas untuk B1, verifikasi tahap A, dan pernyataan jelas apakah tiap mekanisme terkonfirmasi.

**BERHENTI. Tunggu `next`.**

---

# STEP 8. R4 dan R5. Lengan falsifikasi

Tanpa dua lengan ini, reviewer berhak bilang "mungkin modifikasi apa pun juga menolong".

**R4: hipotesis tanding kapasitas.**
Ambil R0b (head nominal), tambahkan blok atensi generik. **Setel ukurannya sampai parameter tambahannya ≥ total parameter B1+B2+B3.** Cetak perbandingan parameternya sebagai bukti. Tanpa penyamaan ini, perbandingannya tidak sah.

*Prediksi tertulis di muka:* Δ ≤ 2σ pada semua metrik, tanpa konsentrasi pada kelas atau jam mana pun.

**R5: isolasi motivasi, bukan modul.**
Karena tahap A beku, versi asli uji ini (P2 ke seluruh head) tidak berlaku. Gantinya, uji **sumber fitur** untuk tahap B:
- stride 4 (yang dipakai B1)
- stride 8
- stride 16

*Prediksi tertulis di muka:* kalau argumen tekstur benar, ada tren monoton yang jelas memihak stride 4. Kalau ketiganya setara, sinyalnya bukan frekuensi tinggi dan argumen tekstur salah.

**Kerjakan:** latih keduanya pada kelima fold, laporkan.

**Kalau R4 mengungguli konfigurasi OSG penuh:** laporkan. Hipotesis paper tidak didukung data, dan itu tetap temuan yang sah.

**Laporkan:** tabel perbandingan parameter untuk R4, tabel tiga sumber fitur untuk R5, kesimpulan eksplisit untuk tiap lengan.

**BERHENTI. Tunggu `next`.**

---

# STEP 9. Statistik akhir

**Kerjakan:**
1. **Seed ganda:** ulangi konfigurasi terbaik dan R0b dengan seed 1 dan 2. Hitung σ antar-seed final untuk semua metrik.
2. **Bootstrap CI:** 1000 resample pada fold uji, selang kepercayaan 95% untuk tiap metrik tiap konfigurasi.
3. **Uji berpasangan:** Wilcoxon signed-rank lintas 5 fold antara R0b dan tiap varian. Laporkan p, dan **nyatakan bahwa n=5 membuat uji ini indikatif, bukan definitif**.
4. **Sensitivitas ambang IoU:** hitung ulang metrik ordinal konfigurasi final pada ambang pencocokan 0,4, 0,5, dan 0,6.
5. **Efisiensi:** parameter, GFLOPs, latensi CPU dan GPU, memori proses, **dilaporkan terpisah untuk tahap A dan tahap B**. Latensi = median 10 pengulangan setelah 3 pemanasan yang tidak dihitung.

**Laporkan:** ringkasan seluruh statistik.

**BERHENTI. Tunggu `next`.**

---

# STEP 10. Gambar

**Kerjakan:**
1. `figures/fig2_confusion.png` : matriks konfusi R0b dan R1 berdampingan, **skala warna sama**, agar pergeseran massa ke diagonal terlihat.
2. `figures/fig3_quartile.png` : bar chart ΔQWK per kuartil jam (uji mekanisme B2). Kalau mekanismenya benar, bentuknya U.
3. `figures/fig4_ablation.png` : ΔQWK tiap konfigurasi dengan error bar σ, plus garis horizontal di 1σ, 2σ, dan 3σ.
4. `figures/roi_sanity_check.png` : sudah ada dari STEP 3, pastikan tersalin.

Semua gambar: 300 DPI minimal, teks terbaca saat dicetak selebar satu kolom, tanpa emoji.

**Laporkan:** tampilkan keempat gambar ke user.

**BERHENTI. Tunggu `next`.**

---

# STEP 11. Rakit bundel hasil dan cek kelengkapan

**Kerjakan:**

1. Pastikan `RESULTS/results.json` memuat, untuk **setiap** konfigurasi (R0, R0b, R1, R2, R3, R4, R5a/b/c) dan **setiap** fold:
   - precision, recall, mAP50, mAP50-95 (keseluruhan dan per kelas)
   - QWK, MAE-ordinal, 1-off accuracy, proporsi lompat ≥2 tingkat
   - matriks konfusi mentah
   - jumlah parameter, GFLOPs
   - waktu latih, epoch terbaik, seed

2. Buat `RESULTS/tables/` berisi tabel **format markdown siap salin**:
   - `tabel3a_satuan_ukur.md` (mean, σ antar-fold, σ antar-seed, nilai 3σ)
   - `tabel3b_R0_vs_R0b.md`
   - `tabel3c_R0b_vs_R1.md`
   - `tabel4_kuartil_jam.md`
   - `tabel5_pasangan_kelas.md` (termasuk baris verifikasi tahap A identik)
   - `tabel6a_kapasitas.md`
   - `tabel6b_sumber_fitur.md`
   - `tabel7_sensitivitas_iou.md`
   - `tabel8_efisiensi.md`

3. Tulis `RESULTS/SUMMARY.md` berisi:
   - Status tiap gerbang (lulus atau gagal, dengan angkanya)
   - Untuk tiap uji mekanisme: terkonfirmasi, tidak konklusif, atau tertolak, beserta angkanya
   - **Daftar eksplisit setiap hal yang tidak berjalan sesuai prediksi**
   - Setiap penyimpangan dari brief ini beserta alasannya
   - Total jam GPU terpakai

4. **Cek kelengkapan otomatis.** Tulis skrip yang memverifikasi:
   - Tidak ada nilai `NOT_YET_COMPUTED` tersisa di `results.json`
   - Tidak ada `NaN` atau `null` di metrik mana pun
   - Kelima fold ada untuk setiap konfigurasi
   - Metrik tahap A **identik persis** di R0b sampai R5 (kalau tidak, tandai sebagai KESALAHAN KRITIS)
   - Keempat gambar ada dan ukurannya bukan nol
   - Kesembilan berkas tabel ada

   Skrip mencetak `LENGKAP` atau daftar yang kurang.

5. Zip seluruh folder `RESULTS/` jadi `RESULTS_YOLO26OSG_<tanggal>.zip`. Sertakan `ENV.txt` dan `RUNLOG.md`. **Jangan** sertakan cache fitur atau bobot model (terlalu besar).

**Laporkan ke user:**
- Hasil cek kelengkapan
- Isi `SUMMARY.md`
- Lokasi berkas zip
- Kalimat penutup: *"Bundel siap dikirim. Kirim `RESULTS_YOLO26OSG_<tanggal>.zip` ke percakapan perancangan untuk verifikasi."*

**SELESAI.**

---

## LAMPIRAN A. Definisi metrik

**Protokol pencocokan.** Prediksi dicocokkan ke ground truth dengan IoU ≥ 0,5 memakai Hungarian matching (`scipy.optimize.linear_sum_assignment`). Hanya pasangan ter-match yang masuk perhitungan metrik ordinal. Deteksi tak ter-match dilaporkan terpisah sebagai FP dan FN, **tidak dicampur** ke metrik ordinal.

**QWK.** `sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')` pada level 0..3.

**MAE-ordinal.** `mean(|level_true - level_pred|)` pada pasangan ter-match.

**1-off accuracy.** Proporsi pasangan dengan `|level_true - level_pred| <= 1`.

**Proporsi lompat ≥2 tingkat.** Proporsi pasangan dengan `|level_true - level_pred| >= 2`. Pada baseline SIMIKA nilainya nol; kalau nilaimu jauh di atas nol, periksa apakah ada yang salah.

## LAMPIRAN B. Yang TIDAK boleh dilakukan

- Jangan menyetel hyperparameter berdasarkan hasil di test fold. Tidak ada seleksi model di test fold.
- Jangan mengubah konfigurasi tahap A antar konfigurasi tahap B. Sekali beku, tetap beku.
- Jangan melaporkan hasil seed terbaik. Laporkan mean dan std lintas seed.
- Jangan melewati gerbang yang gagal.
- Jangan mengarang angka, termasuk sebagai contoh sementara.
- Jangan menambah augmentasi atau trik yang tidak ada di brief ini tanpa memberitahu user.

## LAMPIRAN C. Kalau ada yang gagal

Kalau sebuah STEP gagal karena error teknis (OOM, dependency, dsb): perbaiki, catat di `RUNLOG.md`, ulangi STEP itu, laporkan apa yang terjadi.

Kalau OOM di 8GB VRAM: turunkan batch ke 8, catat perubahannya, dan **pakai batch yang sama untuk semua konfigurasi** agar perbandingannya tetap sah.

Kalau sebuah STEP gagal karena hasilnya tidak sesuai prediksi: **itu bukan kegagalan.** Laporkan apa adanya dan tunggu keputusan user.
