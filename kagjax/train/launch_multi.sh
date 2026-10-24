#!/bin/bash
# Runs on a dedicated multi-GPU machine (operator plan Sep 27: 8x A100 80 GB or 4x H100, 10M ONLY, runs A and B side by
# side; design.md "Runs A and B").  Each plan line starts ONE run (run_gpu.sh) on a GPU list: one GPU = a single-device
# run, several GPUs = one data-parallel run over them (MODE=dp; MODE=seeds on the line = one run per listed GPU).
#
# Usage:  bash kagjax/train/launch_multi.sh <prefix> [plan]
#   plan = ab10m        (default) A on GPUs 0-3 and B on GPUs 4-7, each data parallel over its 4 GPUs (4x H100: A on
#                       0-1, B on 2-3)
#          ab10m_seeds  fallback when the data-parallel probe fails: one run per GPU (A seeds on 0-3, B seeds on 4-7)
#          bsweep       first hour: four from-scratch B variants, one per GPU on 4-7 (learning-rate / entropy / value)
#          mix          the Sep 26 plan (even GPUs 20M, odd GPUs 10M, fast preset)
#          <file>       lines "<gpu[,gpu...]> <run_name> [VAR=value ...]" ('#' lines ignored)
# Each run: kagjax/runs/<prefix>_<run_name>/ (log.txt, eval.jsonl, stats.jsonl, pt/it*.pt every PT_EVERY_SAMPLES =
# 100M learner samples, best.pt, state_*.npz for RESUME=...).  Stop a run: kill its PID (printed below).
R=/Users/asad.ali/kaggriculture
PREFIX=${1:?usage: launch_multi.sh <prefix> [ab10m|ab10m_seeds|bsweep|mix|plan_file]}
PLAN=${2:-ab10m}
NG=$(nvidia-smi -L | wc -l | tr -d ' ')
glist() { seq "$1" "$2" | paste -sd, -; }               # "0,1,2,3" (BSD seq -s adds a trailing comma)
half=$((NG / 2)); A_G=$(glist 0 $((half - 1))); B_G=$(glist $half $((NG - 1)))
TMP=$(mktemp)
case "$PLAN" in
  ab10m)
    echo "$A_G A10m RECIPE=A SEED=1" >> "$TMP"
    echo "$B_G B10m RECIPE=B SEED=2" >> "$TMP" ;;
  ab10m_seeds)
    echo "$A_G A10m RECIPE=A MODE=seeds SEED=100" >> "$TMP"
    echo "$B_G B10m RECIPE=B MODE=seeds SEED=200" >> "$TMP" ;;
  bsweep)                                               # ~45 min, B=2048 per GPU; compare vs PASS / IL@0 learning
    SW="RECIPE=B B=2048 EVAL_EVERY=3 EVAL_N=256 EVAL_OPPS=PASS,IL@0,IL@10 SNAP_EVERY=3"
    VARS=("Bsw_lr1e-4 LR=1e-4" "Bsw_lr3e-4 LR=3e-4" "Bsw_ent03 LR=1e-4 ENT_COEF=0.03" "Bsw_vnet LR=1e-4 VALUE_INIT=policy VALUE_GRAD=0")
    j=0
    for g in ${B_G//,/ }; do                           # one variant per GPU of the B half (4x H100: the first two)
      [ $j -lt ${#VARS[@]} ] || break
      set -- ${VARS[$j]}; nm=$1; shift
      echo "$g $nm $SW SEED=$g $*" >> "$TMP"; j=$((j + 1))
    done ;;
  mix)
    for i in $(seq 0 $((NG - 1))); do
      if [ $((i % 2)) -eq 0 ]; then echo "$i 20m_s$i MODEL=20m SEED=$i B=2048 K=8" >> "$TMP"
      else echo "$i 10m_s$i MODEL=10m SEED=$i B=3072 K=8" >> "$TMP"; fi
    done ;;
  *)
    cp "$PLAN" "$TMP" ;;
esac
echo "plan ($PLAN, $NG GPUs):"; cat "$TMP"
while read -r gpus name rest; do
  if [ -z "$gpus" ] || [[ "$gpus" == \#* ]]; then continue; fi
  gpus=${gpus%,}; n=$(echo "$gpus" | tr ',' '\n' | grep -c .)
  mode=dp; [[ " $rest " == *" MODE=seeds "* ]] && mode=seeds
  # shellcheck disable=SC2086
  if [ "$mode" = seeds ]; then
    CUDA_VISIBLE_DEVICES=$gpus bash "$R/kagjax/train/run_gpu.sh" "${PREFIX}_${name}" $rest
  else
    CUDA_VISIBLE_DEVICES=$gpus DEVICES=$n MODE=dp bash "$R/kagjax/train/run_gpu.sh" "${PREFIX}_${name}" $rest
  fi
done < "$TMP"
rm -f "$TMP"
