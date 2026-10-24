#!/bin/bash
# The Sep 28 sprint on one 8-GPU machine (operator plan: hour 1 = A warm start on all GPUs, hour 2 = B from scratch on all
# GPUs, compare, the winner resumes on all GPUs, the last ~10% win-only).  design.md "Sprint".  From the repo root:
#   bash kagjax/train/sprint.sh preflight                     # ~10 min: tools, data, tests, 2 iterations of each recipe
#   bash kagjax/train/sprint.sh memprobe [2048,3072,4096,5120] [real_B]   # ~3-5 min on GPU 0: games per GPU (>= 15% headroom)
#   bash kagjax/train/sprint.sh A [minutes=60] [games_per_gpu=4096]     # warm start, all GPUs, one data-parallel run
#   bash kagjax/train/sprint.sh B [minutes=60] [games_per_gpu=4096]     # from scratch, all GPUs
#   bash kagjax/train/sprint.sh resume A|B <minutes> [horizon_samples=3e9] [win]   # continue the winner from its last state
#        (win = the win-only phase: MARGIN_W=0 WINB=1, one value-only iteration first)
#   bash kagjax/train/sprint.sh status                        # both runs: iterations, W trends, evaluations, exports
# Every run stops by itself (MAX_MINUTES) with a final evaluation, state_<it>.npz, pt/it<it>.pt and FINAL.json.
# Options: SPRINT_MODE=seeds (one run per GPU instead of data parallel), XLA_EXTRA="<flags>" (optional A/B, see design.md),
# any ppo.py variable (e.g. LR=..., OPPS=...).
R=/Users/asad.ali/kaggriculture
T=$R/kagjax/train
PY=${PY:-/opt/jaxenv/bin/python}
NG=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
ALL=$(seq 0 $((NG - 1)) | paste -sd, -)
cmd=${1:-status}; shift
launch() {   # launch <run> <minutes> <extra VAR=value ...>
  local run=$1 min=$2; shift 2
  if [ "${SPRINT_MODE:-dp}" = seeds ]; then
    CUDA_VISIBLE_DEVICES=$ALL bash $T/run_gpu.sh "$run" MODE=seeds MAX_MINUTES=$min "$@"
  else
    CUDA_VISIBLE_DEVICES=$ALL DEVICES=$NG MODE=dp bash $T/run_gpu.sh "$run" MAX_MINUTES=$min "$@"
  fi
  echo "  log: tail -f ${RUN_ROOT:-$R/kagjax/runs}/$run/log.txt | cut -c1-220   (status: bash $T/sprint.sh status)"
}
case "$cmd" in
  preflight)
    PY=$PY bash $T/preflight.sh ;;
  memprobe)
    CUDA_VISIBLE_DEVICES=0 KAGJAX_BACKEND=real PYTHONPATH=$R:$PYTHONPATH XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
      $PY $T/mem_probe.py --b "${1:-2048,3072,4096,5120}" ${2:+--run $2} ;;
  A)
    launch sprint_A "${1:-60}" RECIPE=A B="${2:-4096}" SEED=1 "${@:3}" ;;
  B)
    launch sprint_B "${1:-60}" RECIPE=B B="${2:-4096}" SEED=2 "${@:3}" ;;
  resume)
    w=${1:?resume A|B <minutes> [horizon] [win]}; min=${2:?minutes}; hor=${3:-3e9}; run=sprint_$w; dir=${RUN_ROOT:-$R/kagjax/runs}/$run
    st=$(ls $dir/state_*.npz 2>/dev/null | sed 's/.*state_\([0-9]*\)\.npz/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$st" ] || { echo "no state in $dir"; exit 1; }
    B_=$($PY -c "import json, numpy as np; print(json.loads(str(np.load('$st')['__meta__']))['cfg']['games'])")
    extra=""; [ "${4:-}" = win ] && extra="MARGIN_W=0 WINB=1 VWARM_RESUME=1"
    echo "resume $run from $st (B=$B_ per GPU, horizon $hor samples, $min min) $extra"
    # shellcheck disable=SC2086
    launch $run "$min" RECIPE=$w B=$B_ SEED=$([ $w = A ] && echo 1 || echo 2) RESUME=$st EVAL_BASE=$dir/eval_0.npz HORIZON_SAMPLES=$hor $extra "${@:5}" ;;
  status)
    PYTHONPATH=$R $PY $T/sprint_status.py ${RUN_ROOT:-$R/kagjax/runs}/sprint_A ${RUN_ROOT:-$R/kagjax/runs}/sprint_B ;;
  *)
    echo "usage: sprint.sh preflight | memprobe | A | B | resume A|B <min> [horizon] [win] | status"; exit 2 ;;
esac
