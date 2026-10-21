#!/usr/bin/env bash
# Pipeline repo ini: generate dataset -> analisis.
# Pakai:  bash run.sh
set -euo pipefail
# Pakai interpreter yang punya numpy/pandas/sklearn. Urutan: $PYTHON, venv lokal,
# lalu python3 sistem. (python3 sistem sering tanpa pandas -> ImportError.)
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python
  elif [ -x /tmp/skripsi-venv/bin/python ]; then PY=/tmp/skripsi-venv/bin/python
  else PY=python3; fi
else
  PY="$PYTHON"
fi
echo "== interpreter: $PY =="
echo "== generate dataset =="
"$PY" src/generate_dataset.py
echo "== analisis =="
"$PY" src/materi6_forecasting.py
echo
echo "Selesai. Hasil ada di outputs/"
