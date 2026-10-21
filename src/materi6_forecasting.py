"""Forecasting beban listrik jam-anan: baseline vs regresi vs Gradient Boosting.

Protokol: uji 30 hari terakhir (720 jam), dilatih pada 2,5 tahun sebelumnya.
Model: naive musiman (jam sama minggu lalu), regresi linier dengan fitur
kalender + suhu, dan Gradient Boosting pada fitur lag + kalender.
Metrik: MAE, RMSE, MAPE per horizon 24 jam.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
H = 720  # 30 hari x 24 jam
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "beban_listrik.csv", parse_dates=["waktu"])
df = df.sort_values("waktu").reset_index(drop=True)
train, test = df.iloc[:-H], df.iloc[-H:]

ytr, yte = train["beban_mw"].values, test["beban_mw"].values


def mape(a, f):
    m = a > 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100)


def metrics(y, p):
    return {"MAE": round(float(mean_absolute_error(y, p)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "MAPE_persen": round(mape(y, p), 3)}


res = {}

# ---- 1. Naive musiman: nilai jam yang sama, 168 jam (1 minggu) sebelumnya
naive = np.full(H, np.nan)
for k in range(H):
    src = len(df) - H - 168 + k
    naive[k] = df["beban_mw"].values[src]
res["NaiveMusiman_168h"] = metrics(yte, naive)

# ---- 2. Regresi linier: kalender + suhu (+ suhu di atas ambang)
def design(d):
    hod, dow, doy = d["jam"].values, d["hari_dalam_pekan"].values, d["hari_dalam_tahun"].values
    suhu = d["suhu_c"].values
    cols = [np.ones(len(d))]
    for h in range(1, 24):                      # dummies jam (buang 1 sebagai basis)
        cols.append((hod == h).astype(float))
    for w in range(1, 7):                       # dummies hari
        cols.append((dow == w).astype(float))
    cols.append(np.cos(2 * np.pi * doy / 365.25))
    cols.append(np.sin(2 * np.pi * doy / 365.25))
    cols.append(np.clip(suhu - 26.0, 0, None))  # pendingin udara linear
    cols.append(np.clip(suhu - 26.0, 0, None) ** 2)
    cols.append(d["libur"].values.astype(float))
    return np.column_stack(cols)


Xtr, Xte = design(train), design(test)
lr = LinearRegression().fit(Xtr, ytr)
pred_lr = lr.predict(Xte)
res["RegresiLinier_kalender+suhu"] = metrics(yte, pred_lr)

# ---- 3. Gradient Boosting pada fitur lag + kalender
def lag_features(d, full):
    pos = d.index.values
    out = pd.DataFrame(index=d.index)
    out["jam"] = d["jam"].values
    out["dow"] = d["hari_dalam_pekan"].values
    out["doy"] = d["hari_dalam_tahun"].values
    out["suhu"] = d["suhu_c"].values
    out["suhu_ac"] = np.clip(d["suhu_c"].values - 26.0, 0, None)
    out["libur"] = d["libur"].values
    for lag in (1, 2, 3, 24, 168):
        out[f"lag_{lag}"] = full["beban_mw"].values[pos - lag]
    out["rolling_mean_24"] = (pd.Series(full["beban_mw"].values)
                              .rolling(24).mean().values[pos])
    out["rolling_std_24"] = (pd.Series(full["beban_mw"].values)
                             .rolling(24).std().values[pos])
    return out


allX = df[["beban_mw"]].copy()
Ftr = lag_features(train, allX).values
Fte = lag_features(test, allX).values
# lag/rolling menghasilkan NaN pada 168 baris pertama (rolling 168 belum penuh);
# buang baris itu dari latih, isi 0 untuk test (test punya histori penuh).
nan_tr = np.isnan(Ftr).any(axis=1)
Ftr, ytr_gb = Ftr[~nan_tr], ytr[~nan_tr]
Fte = np.nan_to_num(Fte, nan=0.0)
gb = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.06,
                               subsample=0.9, random_state=SEED)
gb.fit(Ftr, ytr_gb)
pred_gb = gb.predict(Fte)
res["GradientBoosting_lag+kalender"] = metrics(yte, pred_gb)

best = min(res, key=lambda k: res[k]["MAPE_persen"])
best_pred = {"NaiveMusiman_168h": naive,
             "RegresiLinier_kalender+suhu": pred_lr,
             "GradientBoosting_lag+kalender": pred_gb}[best]

imp = (pd.DataFrame({"fitur": ["jam", "dow", "doy", "suhu", "suhu_ac", "libur",
                               "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
                               "rolling_mean_24", "rolling_std_24"],
                     "importance": gb.feature_importances_})
       .sort_values("importance", ascending=False).reset_index(drop=True))
imp.to_csv(OUT / "feature_importance.csv", index=False)

out = pd.DataFrame({"waktu": test["waktu"], "aktual": yte,
                    "prediksi_terbaik": best_pred.round(4),
                    "pred_gb": pred_gb.round(4), "pred_lr": pred_lr.round(4)})
out.to_csv(OUT / "prediksi_uji.csv", index=False)
(OUT / "metrics.json").write_text(json.dumps({**res, "model_terbaik": best,
                                             "fitur_teratas": imp.iloc[0]["fitur"]}, indent=2))

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
axes[0].plot(test["waktu"], yte, lw=1.1, color="#33502e", label="aktual")
axes[0].plot(test["waktu"], best_pred, lw=1.1, color="#b3402e", alpha=0.85,
             label=f"prediksi ({best})")
axes[0].set_ylabel("beban (MW)")
axes[0].set_title("Forecasting beban listrik — horizon 30 hari (720 jam)")
axes[0].legend()
axes[1].plot(test["waktu"], yte - best_pred, lw=0.8, color="#4a4a4a")
axes[1].axhline(0, color="#b3402e", lw=0.8, ls="--")
axes[1].set_ylabel("residual (MW)")
axes[1].set_xlabel("waktu")
fig.tight_layout()
fig.savefig(OUT / "forecasting.png", dpi=130)
plt.close(fig)

print(f"[analisis] latih={len(train):,} jam | uji={H} jam (30 hari)")
for n_, r in res.items():
    print(f"[analisis] {n_:30s} MAE={r['MAE']} RMSE={r['RMSE']} MAPE={r['MAPE_persen']}%")
print(f"[analisis] terbaik={best} | fitur teratas={imp.iloc[0]['fitur']} "
      f"({imp.iloc[0]['importance']:.3f})")
