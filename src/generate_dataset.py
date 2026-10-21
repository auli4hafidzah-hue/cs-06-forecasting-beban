"""Deret beban listrik jam-anan sintetis 3 tahun (2024-2026) untuk forecasting.

Komponen: tren musiman (mingguan + tahunan), suhu harian sebagai regressor
(efek pendingin udara non-linear), libur nasional, dan shock acak. Ground truth
untuk evaluasi: 30 hari terakhir sebagai horizon uji.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

start = pd.Timestamp("2024-01-01 00:00:00")
end = pd.Timestamp("2026-12-31 23:00:00")
idx = pd.date_range(start, end, freq="h")
n = len(idx)

hod = idx.hour.values
dow = idx.dayofweek.values
doy = idx.dayofyear.values

# profil jam-anan: dua puncak (pagi & malam), dasar malam rendah
jam_profil = (0.62
              + 0.20 * np.exp(-((hod - 9) / 2.6) ** 2)
              + 0.26 * np.exp(-((hod - 20) / 2.9) ** 2))
# akhir pekan turun
pe_effect = np.where(dow >= 5, -0.13, 0.0)
# musiman tahunan: puncak Januari & Juli (pendingin udara)
musim = 0.09 * np.cos(2 * np.pi * (doy - 15) / 365.25)
# tren: permintaan naik ~2,5%/tahun
tren = 0.025 * (doy / 365.25)
# suhu harian: sinus tahunan + noise, satuan derajat C
suhu = (28.0 + 4.5 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        + rng.normal(0, 1.8, n))
# efek non-linear pendingin udara: baru aktif di atas 26 C, kuadratik
ac = 0.011 * np.clip(suhu - 26.0, 0, None) ** 1.9

libur = np.zeros(n, dtype=int)
for d, m in [(1, 1), (17, 8), (25, 12)]:
    libur |= ((idx.month.values == m) & (idx.day.values == d)).astype(int)
libur_effect = -0.16 * libur

shock = rng.normal(0, 0.028, n)
# 8 shock cuaca ekstrem (gelombang panas) durasi 6-20 jam
for _ in range(8):
    p = int(rng.integers(0, n - 30))
    dur = int(rng.integers(6, 21))
    suhu[p:p + dur] += rng.normal(6.5, 1.5)
    ac[p:p + dur] = 0.011 * np.clip(suhu[p:p + dur] - 26.0, 0, None) ** 1.9
    shock[p:p + dur] += rng.normal(0.05, 0.015, dur)

beban = (jam_profil + pe_effect + musim + tren + libur_effect + ac + shock)
beban = np.clip(beban, 0.25, None)

df = pd.DataFrame({"waktu": idx, "beban_mw": beban.round(4),
                   "suhu_c": suhu.round(2), "libur": libur,
                   "jam": hod, "hari_dalam_pekan": dow, "hari_dalam_tahun": doy})
df.to_csv(DATA / "beban_listrik.csv", index=False)
print(f"[generate] beban_listrik.csv: {len(df):,} baris jam-anan")
print(f"[generate] rentang {df['waktu'].min()} s/d {df['waktu'].max()}")
print(f"[generate] beban rata-rata {df['beban_mw'].mean():.3f} MW | "
      f"min {df['beban_mw'].min():.3f} | max {df['beban_mw'].max():.3f} MW")
print(f"[generate] gelombang panas disuntikkan: 8 kejadian")
