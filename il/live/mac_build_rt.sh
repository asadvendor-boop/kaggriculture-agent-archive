#!/bin/bash
# Sep 29: build the R (repeat opponents, last 3 days) and T (top-10 leaders, last day) datasets - winners' moves only, team 10, games already in A skipped.
K=/Users/asad.ali/kaggriculture; W=$K/il/ilwin; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd $K
LOG=$K/il/logs/win29_rt.log; F="--min-opp-rating 2400 --min-margin 500 --fail-mask 1 --strip-churn 1"
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG; }
build() {
  out=$1; nm=$2; shift 2; tag=$(basename $out)
  say "build $tag: $*"
  for i in 0 1; do
    TORCH_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $PY $W/build_winner_ds.py --out $out --tapes "$@" $F --winner-names $nm \
      --exclude-manifest $K/il/data/win29/manifest.json --worker $i --workers 2 > $K/il/logs/win29_${tag}_w$i.log 2>&1 &
  done
  wait
  grep -h "selected\|per dir" $K/il/logs/win29_${tag}_w0.log | head -2 | tee -a $LOG
  grep -h "done:" $K/il/logs/win29_${tag}_w*.log | tee -a $LOG
  $PY $W/build_winner_ds.py --out $out --merge > $K/il/logs/win29_${tag}_merge.log 2>&1
  $PY $W/check_ds.py $out 2>&1 | tee -a $LOG
}
build $K/il/data/win29r $K/il/live/rep29_names.json data/tapes_rep29:L data/tapes_all:L data/tapes_allg:L data/tapes_st2:L data/tapes_soh:L data/tapes_n20mw8:L
say "R READY"
build $K/il/data/win29t $K/il/live/top10_names.json data/tapes_top10d:L
say "T READY"
