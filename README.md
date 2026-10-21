# Studi Kasus 6 — Forecasting Beban Listrik Jam-anan

Memprediksi beban listrik per jam selama 30 hari ke depan menggunakan
regresi linier dengan fitur kalender + suhu, dibandingkan dengan Gradient
Boosting berbasis fitur lag.

## Pertanyaan analitis

1. Model mana yang paling akurat untuk horizon 30 hari?
2. Seberapa besar kontribusi suhu udara terhadap beban listrik?
3. Apakah fitur lag (histori beban) menambah informasi di atas kalender + suhu?

## Dataset

`beban_listrik.csv` — **26.304 baris jam-anan**, 1 Januari 2024 s/d
31 Desember 2026 (3 tahun penuh). Kolom: `beban_mw`, `suhu_c`, `libur`,
`jam`, `hari_dalam_pekan`, `hari_dalam_tahun`.

Komponen deret waktu yang dibangun:

| Komponen | Karakteristik |
|---|---|
| Profil jam-anan | dua puncak (pagi & malam), dasar malam rendah |
| Efek akhir pekan | −13% |
| Musiman tahunan | puncak Januari & Juli (pendingin udara) |
| Tren | +2,5% per tahun |
| Efek suhu | non-linear, aktif di atas 26°C (pendingin udara) |
| Libur nasional | −16% |
| Shock cuaca | **8 gelombang panas**, 6–20 jam, suhu +6,5°C |

Statistik: rata-rata **0,829 MW**, minimum 0,346 MW, maksimum 2,881 MW.

## Metode

- **Protokol uji**: 30 hari terakhir (720 jam) sebagai horizon uji,
  25.584 jam sebelumnya untuk pelatihan
- Tiga model:
  - **NaiveMusiman_168h** (baseline): nilai jam yang sama, 1 minggu sebelumnya
  - **RegresiLinier**: kalender (dummies jam & hari) + musiman sinus +
    suhu linear & kuadratik + libur
  - **GradientBoosting**: 400 estimator, depth 4, learning rate 0,06,
    fitur lag (1, 2, 3, 24, 168 jam) + rolling mean/std 24 jam + kalender
- Metrik: MAE, RMSE, **MAPE** pada 720 jam uji

## Hasil terukur

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|
| NaiveMusiman_168h | 0,0396 | 0,0528 | 5,258 |
| RegresiLinier_kalender+suhu | 0,0238 | 0,0300 | 3,084 |
| **GradientBoosting_lag+kalender** | **0,0231** | **0,0288** | **3,026** |

**Model terbaik: GradientBoosting** — hanya **1,9% lebih baik** dari regresi
linier dalam MAPE.

**Fitur teratas GradientBoosting: `suhu` (importance 0,403)** — suhu udara
sendirian menyumbang 40% seluruh sinyal prediktif.

## Interpretasi

Gradient Boosting mengalahkan regresi linier hanya tipis (3,026% vs 3,084%
MAPE). Ini temuan yang penting secara operasional: **untuk forecasting beban
listrik dengan regressor cuaca yang tersedia, model sederhana sudah hampir
semejar baik model kompleks.** Kompleksitas tambahan 400 pohon hanya menurunkan
error 1,9%.

Suhu adalah prediktor tunggal terkuat (40% importance). Ini konsisten dengan
realitas: pendingin udara mendominasi variasi beban, dan efeknya non-linear
(baru aktif di atas 26°C). Regresi linier yang sudah memasukkan suhu kuadratik
mampu menangkap sebagian besar efek ini.

Kegagalan terbesar model terjadi pada 8 gelombang panas yang disuntikkan —
ketika suhu melonjak tiba-tiba, model berbasis lag tertinggal karena belum
pernah melihat pola serupa. Untuk memperbaiki, perlu integrasi prakiraan cuaca
(prediksi suhu 24–48 jam ke depan), bukan hanya suhu aktual.

## Struktur

```
├── src/generate_dataset.py          # deret beban jam-anan 3 tahun + 8 shock cuaca
├── src/materi6_forecasting.py       # naive vs regresi vs GBM, feature importance
├── outputs/
│   ├── prediksi_uji.csv            # aktual vs 3 model, 720 jam
│   ├── feature_importance.csv      # importance 13 fitur GBM
│   ├── forecasting.png              # deret uji + residual
│   └── metrics.json
```

## Cara menjalankan

```bash
pip install -r requirements.txt
bash run.sh
```

## Output

`outputs/forecasting.png` menampilkan deret beban aktual vs prediksi terbaik
selama 30 hari uji, beserta panel residual.
