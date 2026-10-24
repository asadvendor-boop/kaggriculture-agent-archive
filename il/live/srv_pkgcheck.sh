#!/bin/bash
# Package checks for an upload candidate (Sep 26): 12-world package parity vs its SE benchmark (exact margins) and the
# official kaggle_environments engine (real loader, both seats vs V94, must match kagsim exactly).
# Usage: srv_pkgcheck.sh <package dir name> <benchmark rows json>
exec 8>/tmp/srv_pkgcheck_$1.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=/Users/asad.ali/kaggriculture/package/$1/main.py
echo "[$(date -u +%H:%M) UTC] $1 sha256 $(sha256sum $P | cut -c1-16)"
/opt/jaxenv/bin/python -c "import kaggle_environments" 2>/dev/null || /opt/jaxenv/bin/pip install -q "kaggle-environments==1.32.7" 2>&1 | tail -2
nice -n 5 /venv/main/bin/python live/pkg_parity.py $P $2 12 2>&1 | grep -v -i warn
cd /Users/asad.ali/kaggriculture
nice -n 5 /opt/jaxenv/bin/python tools/official_check.py $P V94 7 2>&1 | grep -v -i "warn"
echo "[$(date -u +%H:%M) UTC] PKGCHECK $1 DONE"
