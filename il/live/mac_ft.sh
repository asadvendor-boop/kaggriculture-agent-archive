#!/bin/bash
# Sep 29 winner-imitation fine-tune on the Mac M5 GPU (mps). ONE process at a time on the GPU.
# Usage: bash il/live/mac_ft.sh <tag> "<seat dirs, colon-separated>" "<x6 dirs>" <weights.json> [PASSES] [MAX_LR]
#   e.g.  bash il/live/mac_ft.sh A il/data/win29/seats il/data/win29/x6 il/data/win29/win_weights.json
# Writes il/ckpt/ilsp_v13_win_<tag>.pt (+ _h1.._hN half-pass checkpoints), log il/logs/winft_<tag>.log, summary il/logs/winft_mac.log
K=/Users/asad.ali/kaggriculture; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd $K
TAG=$1; SEATS=$2; X6=$3; WW=$4; PASSES=${5:-2}; LR=${6:-5e-5}
export OMP_NUM_THREADS=1
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a il/logs/winft_mac.log; }
[ -f $K/il/live/SKIP_$TAG ] && { say "SKIP $TAG (operator changed the plan)"; exit 1; }
say "train $TAG: seats $SEATS | passes $PASSES lr $LR"
run_train() {
  env DEVICE=mps SEED=0 MAX_LR=$LR INIT=il/ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=50 VAL_MAX=180 VAL_STATES=20000 PROGRESS_EVERY=25 AUX_W=0.1 \
      LOAD_THREADS=2 PREFETCH=1 WIN_DIRS=$SEATS X6_DIRS=$X6:il/data/x6 WIN_W=$WW REH="il/data/fresh2:1.0,il/data/dsm_ft2:0.5" \
      WIN_TEAM_MODE=ours SAVE_EVERY=0.5 nice -n 10 $PY il/ilwin/train_winner.py il/ckpt/ilsp_v13_win_$TAG.pt $PASSES 320 > il/logs/winft_$TAG.log 2>&1 &
  TP=$!; t0=$(date +%s); last=$t0
  while kill -0 $TP 2>/dev/null; do                      # stall watchdog: the M5 GPU hung once (Sep 29) inside nonzero_mps -> no log line for 25 min
    sleep 30; now=$(date +%s)
    if [ $(( now - last )) -gt 120 ]; then say "system sleep detected ($(( now - last ))s) - watchdog timers reset"; touch il/logs/winft_$TAG.log; t0=$now; fi
    last=$now
    age=$(( $(date +%s) - $(stat -f %m il/logs/winft_$TAG.log) )); up=$(( $(date +%s) - t0 ))
    if [ $age -gt 420 ] && [ $up -gt 420 ]; then say "STALL: no log output for ${age}s -> kill -9 $TP"; kill -9 $TP; sleep 3; return 1; fi
  done
  wait $TP
}
for attempt in 1 2 3; do run_train && break; say "attempt $attempt failed/stalled"; rm -f il/ckpt/ilsp_v13_win_${TAG}*.pt; done
grep -E "^\[|saved|per pass|winner seats" il/logs/winft_$TAG.log | tail -14 | tee -a il/logs/winft_mac.log
[ -f il/ckpt/ilsp_v13_win_$TAG.pt ] || { say "TRAINING FAILED $TAG: $(tail -3 il/logs/winft_$TAG.log)"; exit 1; }
for c in $(ls il/ckpt/ilsp_v13_win_${TAG}_h*.pt | sort -V); do
  DEVICE=cpu TORCH_THREADS=1 nice -n 10 $PY il/ilwin/agree_check.py $c --set win=${SEATS%%:*} --set reh=il/data/fresh2:il/data/dsm_ft2 --x6 ${X6%%:*}:il/data/x6 --n 1500 \
      --out ${c%.pt}_agree.json > ${c%.pt}_agree.log 2>&1
  { echo "== agree $(basename $c)"; grep -E "^\[|^   " ${c%.pt}_agree.log | head -12; } | tee -a il/logs/winft_mac.log
done
say "DONE $TAG -> $(ls il/ckpt/ilsp_v13_win_${TAG}*.pt | grep -v agree | xargs -n1 basename | tr '\n' ' ')"
