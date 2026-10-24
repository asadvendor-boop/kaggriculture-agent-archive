#!/bin/bash
# The GPU is idle after tonight's runs (RL is CPU-bound): more seeds of the WINNING recipe (fresh 4M pass, 4 x data/fresh
# at 5e-4 from the grown copy of bc_v6, then DSM fine-tune 2 x data/dsm_ft at 2e-4; seeds 0/1 = 420/414 wins), seeds 2
# and 3 - a seed lottery on a proven recipe - plus one cheap variant: the old-menu fine-tune on DSM + DECEM together
# (data/dsmdecem8: 608 + 605 fresh seats) from the seed-0 fresh pass. 464 ladder benchmarks (CPU governor).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
run() {  # $1 seed
  env SEED=$1 BLOCKS=6 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/grow_v6_176x6.pt \
    $PY train6.py ckpt/fr_g_s$1.pt 4 176 data/fresh > logs/train_fr_g_s$1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] fresh pass done g s$1: $(grep 'best val loss' logs/train_fr_g_s$1.log)"
  env SEED=$1 BLOCKS=6 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/fr_g_s$1.pt \
    $PY train6.py ckpt/fr_g_dsm_s$1.pt 2 176 data/dsm_ft > logs/train_fr_g_dsm_s$1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] DSM FT done g s$1: $(grep 'best val loss' logs/train_fr_g_dsm_s$1.log)"
}
dd() {
  env SEED=0 BLOCKS=6 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/fr_g_s0.pt \
    $PY train6.py ckpt/fr_g_dd_s0.pt 2 176 data/dsmdecem8 > logs/train_fr_g_dd_s0.log 2>&1
  echo "[$(date -u +%H:%M) UTC] DSM+DECEM FT (old menu) done g s0: $(grep 'best val loss' logs/train_fr_g_dd_s0.log)"
}
run 2 & sleep 60; run 3 & sleep 60; dd
bench_gov ladder ckpt/fr_g_dd_s0.pt:10:0.5 ckpt/fr_g_dd_s0.pt:3:0.5
wait
bench_gov ladder ckpt/fr_g_dsm_s2.pt:10:0.5 ckpt/fr_g_dsm_s3.pt:10:0.5
echo "[$(date -u +%H:%M) UTC] MORE SEEDS DONE"
