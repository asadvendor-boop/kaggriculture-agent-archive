#!/bin/bash
# GPU verification batch A (each command takes the shared GPU lock separately via srv_jax.sh --gpu)
cd /Users/asad.ali/kaggriculture/kagjax
P=/opt/jaxenv/bin/python
echo "=== rng_stats $(date -u +%H:%M:%S)"
bash srv_jax.sh --gpu $P rng_stats.py --n 4096 --cfg gpu --out logs_server/rng_stats_gpu_4096.json
for c in gpu seq par; do
  echo "=== batchindep $c $(date -u +%H:%M:%S)"
  bash srv_jax.sh --gpu $P gpu_verify.py batchindep --data logs_server/ds_tapes_mac64.npz --k 7 --sizes 1,7,4096 --cfg $c --out logs_server/batchindep_gpu_$c.json
done
echo "=== done $(date -u +%H:%M:%S)"
