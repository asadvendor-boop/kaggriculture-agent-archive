#!/bin/bash
# Winner-imitation fine-tune of v13_p3 on the GPU training box (Sep 29). One command = build the winner data (once, cached),
# then one fine-tune + agreement checks. Everything logs to /root/winft.log (summary) and $K/il/logs/winft_<tag>*.log.
#
#   bash finetune_winner.sh <tag> [VAR=value ...]        e.g.  bash finetune_winner.sh a MAX_LR=5e-5 PASSES=2
#
# Defaults = the recommended run (RUNBOOK.md): team 10 for every winner seat, LR 5e-5, 2 passes over the winner seats,
# rehearsal per pass = 1.0 x fresh2 + 0.5 x dsm_ft2 seats (the winner data is then ~40% of the samples), checkpoints every
# half pass (_h1.._h4). Output checkpoints: $K/il/ckpt/ilsp_v13_win_<tag>.pt + _h1.._h4.pt (same format as train6.py).
set -u
K=/Users/asad.ali/kaggriculture; W=$(cd "$(dirname "$0")" && pwd); PY=${PY:-/venv/main/bin/python}; LOG=/root/winft.log
TAG=$1; shift
for kv in "$@"; do export "$kv"; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DS=${DS:-$K/il/data/win29}                                   # winner dataset dir (seats/, x6/, manifest.json)
TAPES=${TAPES:-"data/tapes_st2:WL data/tapes_soh:WL data/tapes_n20mw8:WL data/tapes_npse:L data/tapes_nps:L data/tapes_n20mw5:L"}
BUILD_FLAGS=${BUILD_FLAGS:-"--min-opp-rating 2400 --min-margin 500 --se-label drop --fail-mask 1 --strip-churn 1 --verify 1"}
NWB=${NWB:-8}                                                # build workers (each ~0.5 GB RAM + a CUDA context for the policy checks)
BUILD_DEVICE=${BUILD_DEVICE:-cuda}
MAX_LR=${MAX_LR:-5e-5}; PASSES=${PASSES:-2}; SEED=${SEED:-0}; SAVE_EVERY=${SAVE_EVERY:-0.5}
WIN_TEAM_MODE=${WIN_TEAM_MODE:-ours}; N_TEAMS=${N_TEAMS:-48}  # WIN_TEAM_MODE=true needs N_TEAMS=49 (generic id 48)
REH=${REH:-"$K/il/data/reh_fresh2:1.0,$K/il/data/reh_dsm2:0.5"}
INIT=${INIT:-$K/il/ckpt/ilsp_v13_p3.pt}
CK=$K/il/ckpt/ilsp_v13_win_$TAG.pt
say() { echo "[$(date -u +%H:%M) UTC] $*" | tee -a $LOG; }
mkdir -p $K/il/logs

# ---- 1. winner data (built once; delete $DS to rebuild with other filters)
if [ ! -f $DS/manifest.json ]; then
  say "build $DS from: $TAPES | $BUILD_FLAGS | $NWB workers on $BUILD_DEVICE"
  $PY $W/build_winner_ds.py --out $DS --tapes $TAPES $BUILD_FLAGS --list-only 2>&1 | tee -a $LOG
  for i in $(seq 0 $((NWB - 1))); do
    BUILD_DEVICE=$BUILD_DEVICE nice -n 5 $PY $W/build_winner_ds.py --out $DS --tapes $TAPES $BUILD_FLAGS --device $BUILD_DEVICE \
      --worker $i --workers $NWB > $K/il/logs/winft_build_w$i.log 2>&1 &
  done
  wait
  grep -h "done:" $K/il/logs/winft_build_w*.log | tee -a $LOG
  echo "build failures: $(cat $K/il/logs/winft_build_w*.log | grep -c -E 'FAIL|Traceback')" | tee -a $LOG
  $PY $W/build_winner_ds.py --out $DS --merge ${MERGE_FLAGS:-} > $K/il/logs/winft_merge.log 2>&1
  head -c 3000 $K/il/logs/winft_merge.log | tee -a $LOG
fi
[ -f $DS/manifest.json ] || { say "NO DATASET - stop"; exit 1; }

# ---- 2. fine-tune
say "fine-tune $TAG: INIT $(basename $INIT) MAX_LR $MAX_LR PASSES $PASSES team $WIN_TEAM_MODE N_TEAMS $N_TEAMS REH $REH SEED $SEED"
env SEED=$SEED MAX_LR=$MAX_LR INIT=$INIT BLOCKS=10 N_TEAMS=$N_TEAMS MK_NZ_W=3 LOSS_W=0.7 CHUNK=${CHUNK:-100} VAL_MAX=180 AUX_W=0.1 \
    LOAD_THREADS=4 PREFETCH=1 WIN_DIRS=$DS/seats X6_DIRS=$DS/x6:$K/il/data/x6 WIN_W=$DS/win_weights.json REH="$REH" \
    WIN_TEAM_MODE=$WIN_TEAM_MODE SAVE_EVERY=$SAVE_EVERY ${RATING_W:+RATING_W=$RATING_W} \
    $PY $W/train_winner.py $CK $PASSES 320 > $K/il/logs/winft_$TAG.log 2>&1
grep -E "^\[|saved|per pass|winner seats" $K/il/logs/winft_$TAG.log | tee -a $LOG
[ -f $CK ] || { say "TRAINING FAILED: $(tail -3 $K/il/logs/winft_$TAG.log)"; exit 1; }

# ---- 3. drift gauge for every saved checkpoint (winner val games + v13_p3's own leader val games)
CAL=$K/il/ckpt/ilsp_v13_newplan.pt                          # calibration (once): the plan-changing fine-tune of Sep 28
if [ -f $CAL ] && [ ! -f $K/il/logs/winft_calib_newplan_agree.log ]; then  # (1 epoch @1e-4; lost tomatoes, 111 vs 131 fresh3)
  $PY $W/agree_check.py $CAL --set win=$DS/seats --set reh=$K/il/data/reh_fresh2:$K/il/data/reh_dsm2 --x6 $DS/x6 --n 20000 \
      --out $K/il/logs/winft_calib_newplan_agree.json > $K/il/logs/winft_calib_newplan_agree.log 2>&1
  { echo "== CALIBRATION: newplan vs v13_p3 (a drift level that was too much)"; grep -E "^\[|^   " $K/il/logs/winft_calib_newplan_agree.log; } | tee -a $LOG
fi
for c in $(ls ${CK%.pt}_h*.pt | sort -V); do
  $PY $W/agree_check.py $c --set win=$DS/seats --set reh=$K/il/data/reh_fresh2:$K/il/data/reh_dsm2 --x6 $DS/x6 --n 20000 \
      --out ${c%.pt}_agree.json > ${c%.pt}_agree.log 2>&1
  { echo "== agree $(basename $c)"; grep -E "^\[|^   " ${c%.pt}_agree.log; } | tee -a $LOG
done
say "DONE $TAG -> $(ls ${CK%.pt}*.pt | xargs -n1 basename | tr '\n' ' ')"
