#!/bin/bash
# Kaggle's official daily top-games datasets for the days our own downloads missed (operator OK, Sep 25).
cd /Users/asad.ali/kaggriculture
for d in 2026-09-19 2026-09-20 2026-09-22; do
  nice -n 10 .venv/bin/python il/live/import_kaggle_day.py $d
done
echo "[$(date +%H:%M)] days imported"
