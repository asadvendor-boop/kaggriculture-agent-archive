#!/bin/bash
# First hour on the new machine: does ONE run train on several GPUs together (MODE=dp, shard_map data parallel)?
# Never run on real multi-GPU hardware before (tested on 2 CPU devices only).  ~12-15 min, blocking.
#   bash kagjax/train/dp_probe.sh [gpu_list=0,1,2,3]
# 1. the A recipe on ONE GPU of the list, B=1024 games, 3 iterations (the first compiles), no evaluation;
# 2. the same per GPU on ALL listed GPUs as one data-parallel run (4x the games per iteration);
# Verdict "DP OK" when the dp run is finite, its parameter replicas stay identical ([dp] lines, max |diff| == 0) and
# its samples/s at iteration 3 are >= 3x the single-GPU run's (4 GPUs); else "DP FAIL -> launch_multi.sh <p> ab10m_seeds".
R=/Users/asad.ali/kaggriculture
GPUS=${1:-0,1,2,3}
GPUS=${GPUS%,}; N=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
COMMON="RECIPE=A B=1024 K=4 ITERS=3 EVAL_EVERY=0 EVAL_EVERY_MIN=0 CKPT_EVERY=1000 SNAP_EVERY=1000 PT_EVERY_SAMPLES=0 PT_EVERY_MIN=0"
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 10; done; }
rate() { grep -E "^it 3 " "$1" | sed -E 's/.*; ([0-9,]+) samples\/s.*/\1/' | tr -d ','; }
run() {   # run <name> <gpus> <devices>
  out=$(CUDA_VISIBLE_DEVICES=$2 DEVICES=$3 MODE=dp bash $R/kagjax/train/run_gpu.sh "$1" $COMMON)
  echo "  $out"; wait_pid "$(echo "$out" | grep -o 'PID [0-9]*' | awk '{print $2}')"
  grep -E "^it [0-9]|^\[dp\]|STOP|Error" $R/kagjax/runs/$1/log.txt | cut -c1-160; tail -2 $R/kagjax/runs/$1/stdout.log | cut -c1-200
}
echo "[$(date -u +%H:%M:%S) UTC] 1 GPU (${GPUS%%,*})"; run dp_probe_1 "${GPUS%%,*}" 1
echo "[$(date -u +%H:%M:%S) UTC] $N GPUs ($GPUS), one data-parallel run"; run dp_probe_$N "$GPUS" "$N"
r1=$(rate $R/kagjax/runs/dp_probe_1/log.txt); rn=$(rate $R/kagjax/runs/dp_probe_$N/log.txt)
fin=$(grep -c "STOP" $R/kagjax/runs/dp_probe_$N/log.txt)
bad=$(grep "^\[dp\]" $R/kagjax/runs/dp_probe_$N/log.txt | awk '{print $NF}' | grep -vc '^0$')
nit=$(grep -c "^it 3 " $R/kagjax/runs/dp_probe_$N/log.txt)
echo "samples/s at iteration 3: 1 GPU ${r1:-?}, $N GPUs ${rn:-?}; replica mismatches $bad; stops $fin"
if [ "$nit" = 1 ] && [ "$fin" = 0 ] && [ "$bad" = 0 ] && [ -n "$r1" ] && [ -n "$rn" ] && [ "$rn" -ge $(( r1 * (N - 1) )) ]; then
  echo "DP OK -> bash kagjax/train/launch_multi.sh sep28 ab10m"
else
  echo "DP FAIL -> bash kagjax/train/launch_multi.sh sep28 ab10m_seeds"
fi
