#!/bin/bash
# Sep 27 ~03:00 UTC (operator): the A100 is free now -> measure the JAX trainer's REAL dedicated-GPU speed with the fast
# preset before renting the multi-GPU machine. Back to back, nothing else on the GPU: 20M (g20_w_dsm_s0) at B=2048 K=8,
# then 10M (rf_w_dsm_s0) at B=3072 K=8; 3 iterations each (the first includes compilation), no evaluations.
exec 8>/tmp/srv_dedicated_probe.lock; flock -n 8 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
for spec in "ded_20m MODEL=20m B=2048 K=8" "ded_10m MODEL=10m B=3072 K=8"; do
  set -- $spec; name=$1; shift
  echo "[$(date -u +%H:%M:%S) UTC] $name starts ($*)"
  out=$(bash $R/kagjax/train/run_gpu.sh $name "$@" ITERS=3 EVAL_EVERY=0 PT_EVERY_MIN=100000 CKPT_EVERY=1000 SNAP_EVERY=1000)
  echo "  $out"; pid=$(echo "$out" | grep -o 'PID [0-9]*' | awk '{print $2}')
  while kill -0 $pid 2>/dev/null; do sleep 20; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $R/kagjax/runs/$name/gpu_util.log; done
  echo "[$(date -u +%H:%M:%S) UTC] $name finished"
  grep -E "^it [0-9]" $R/kagjax/runs/$name/log.txt | cut -c1-200
  tail -3 $R/kagjax/runs/$name/stdout.log | cut -c1-200
  awk -F', ' '{g+=$1; n++; if ($2+0 > m) m=$2+0} END {if (n) printf "  GPU util mean %.0f%%, peak memory %d MiB\n", g/n, m}' $R/kagjax/runs/$name/gpu_util.log
done
echo "[$(date -u +%H:%M:%S) UTC] DEDICATED PROBE DONE"
