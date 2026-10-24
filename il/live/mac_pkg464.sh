#!/bin/bash
# usage: mac_pkg464.sh <package dir name> <tag> [workers=2]   (CPU only, nice 15, OMP 1)
cd /Users/asad.ali/kaggriculture/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; N=${3:-2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in $(seq 0 $((N - 1))); do nice -n 15 $PY live/pkg464.py /Users/asad.ali/kaggriculture/package/$1/main.py $2 $k $N > logs/pkg464_$2_w$k.log 2>&1 & done
wait; $PY live/pkg464.py --merge $2 $N
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] PKG464 $2 done: $(python3 -c "import json;r=json.load(open('logs/pkg464_$2.json'));print(len(r),'worlds, wins',sum(x['margin']>0 for x in r),'mean margin',round(sum(x['margin'] for x in r)/len(r)))")" | tee -a logs/night29_summary.log
