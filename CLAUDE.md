## Eksperimen CommIT (experiments/commit-osg/)

Riset untuk publikasi jurnal CommIT. Brief lengkapnya di
`experiments/commit-osg/BRIEF.md`. Baca berkas itu sebelum
mengerjakan apa pun di folder tersebut.

### Aturan yang tidak boleh dilanggar

1. **Satu STEP per giliran.** Kerjakan satu STEP dari BRIEF.md
   sampai tuntas, tulis laporannya, lalu BERHENTI dan tunggu user
   mengetik "next". Jangan pernah menyambung sendiri.

2. **Jangan pernah mengarang angka.** Tidak ada nilai placeholder,
   tidak ada contoh, tidak ada perkiraan. Kalau sebuah angka belum
   dihitung, tulis `NOT_YET_COMPUTED`.

3. **Gerbang itu mengikat.** STEP 2, 3, dan 6 punya kriteria lulus.
   Kalau gagal: berhenti dan laporkan. Jangan longgarkan ambangnya,
   jangan cari jalan memutar, jangan lanjut diam-diam.

4. **Hasil negatif dilaporkan apa adanya.** Modul yang tidak
   menolong, atau lengan pembanding yang menang, adalah temuan yang
   sah. Jangan disembunyikan, jangan dipoles.

5. **Jangan sentuh apa pun di luar `experiments/commit-osg/`.**
   Kode di `scripts/`, `config.yaml`, dan `research/` dipakai naskah
   SIMIKA yang sedang direview dan harus tetap bisa direproduksi apa
   adanya. Boleh dibaca dan dipakai ulang, tidak boleh diubah.

6. **Catat semuanya** ke `experiments/commit-osg/RESULTS/RUNLOG.md`:
   perintah, waktu mulai dan selesai, versi paket, seed, kejadian tak
   terduga, dan setiap penyimpangan dari BRIEF.md beserta alasannya.

7. **Simpan inkremental.** Setiap STEP menulis hasilnya ke disk
   sebelum selesai. Commit di akhir tiap STEP dengan pesan
   `STEP n: <ringkasan>`.

8. Kalau ada yang ambigu, **tanya user**, jangan menebak.

### Catatan Windows

- Ultralytics dataloader sering menggantung dengan `workers > 0`.
  Set `workers=0`, catat pilihannya.
- Bungkus entry point training dengan `if __name__ == "__main__":`
  karena Windows memakai spawn, bukan fork.
- Pakai path absolut.

### Melanjutkan setelah sesi terputus

Baca `BRIEF.md` dan `RESULTS/RUNLOG.md`, tentukan STEP terakhir yang
selesai, lalu lanjutkan dari STEP berikutnya. Jangan mengulang STEP
yang sudah tercatat selesai.