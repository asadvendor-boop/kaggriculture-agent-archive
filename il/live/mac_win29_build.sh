#!/bin/bash
# Sep 29 (operator: "build the data first, don't train yet"): winner-imitation datasets on the Mac, 3 workers x 1 thread, nice 10.
#   il/data/win29   = A  (v13_p3-family wins ST2/SOH/20MW8/ALL/ALLG + losses of all 7 scripts + 20MW5 losses)
#   il/data/win29np = new-plan net's wins (NPSE, NPS) -> variant A7
#   il/data/win29l  = top leaders' games since Sep 28 12:00 UTC (winner's moves; fresh4's 150 games held out) -> variant AL
# Usage: bash il/live/mac_win29_build.sh <fetch pid>      log il/logs/win29_build.log
K=/Users/asad.ali/kaggriculture; W=$K/il/ilwin; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
LOG=$K/il/logs/win29_build.log; FPID=${1:-0}
F="--min-opp-rating 2400 --min-margin 500 --se-label drop --fail-mask 1 --strip-churn 1 --verify 1"
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG; }
build() {
  out=$1; shift; tag=$(basename $out)
  say "build $tag: $*"
  for i in 0 1 2; do
    TORCH_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $PY $W/build_winner_ds.py --out $out --tapes "$@" $F --worker $i --workers 3 \
      > $K/il/logs/win29_${tag}_w$i.log 2>&1 &
  done
  wait
  grep -h "done:" $K/il/logs/win29_${tag}_w*.log | tee -a $LOG
  say "$tag failures: $(cat $K/il/logs/win29_${tag}_w*.log | grep -c -E 'FAIL|Traceback')"
  $PY $W/build_winner_ds.py --out $out --merge > $K/il/logs/win29_${tag}_merge.log 2>&1
  head -c 1800 $K/il/logs/win29_${tag}_merge.log >> $LOG; echo >> $LOG
}
cd $K
build $K/il/data/win29 data/tapes_st2:WL data/tapes_soh:WL data/tapes_n20mw8:WL data/tapes_all:WL data/tapes_allg:WL \
      data/tapes_npse:L data/tapes_nps:L data/tapes_n20mw5:L
say "CORE READY (A)"
build $K/il/data/win29np data/tapes_npse:W data/tapes_nps:W
say "NP-WINS READY (A7 extra)"
for i in $(seq 1 120); do                       # wait for the leader fetch (<= 60 min): FRESH4 READY or the fetch process gone
  grep -q "FRESH4 READY" $K/il/logs/fetch_lead29.log && break
  [ "$FPID" != "0" ] && ! kill -0 $FPID 2>/dev/null && break
  sleep 30
done
say "fetch: $(tail -1 $K/il/logs/fetch_lead29.log)"
build $K/il/data/win29l data/tapes_lead29:L
say "ALL DATASETS READY"
