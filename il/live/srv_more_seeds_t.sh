#!/bin/bash
# Idle GPU (Sep 26 ~02:15 PKT): seed lottery for the size that gave the best single model - fresh 10M pass (4 x data/fresh
# at 5e-4 from grow_v6_256x8) + DSM fine-tune (2 x data/dsm_ft at 2e-4); seeds 2 and 3 (seeds 0/1 gave 429/389 wins).
# 464 ladder benchmarks as DSM (team 10) through the CPU governor.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
run() {  # $1 seed
  env SEED=$1 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/grow_v6_256x8.pt \
    $PY train6.py ckpt/fr_t_s$1.pt 4 256 data/fresh > logs/train_fr_t_s$1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] fresh pass done t s$1: $(grep 'best val loss' logs/train_fr_t_s$1.log)"
  env SEED=$1 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/fr_t_s$1.pt \
    $PY train6.py ckpt/fr_t_dsm_s$1.pt 2 256 data/dsm_ft > logs/train_fr_t_dsm_s$1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] DSM FT done t s$1: $(grep 'best val loss' logs/train_fr_t_dsm_s$1.log)"
}
run 2 & sleep 60; run 3 & wait
bench_gov ladder ckpt/fr_t_dsm_s2.pt:10:0.5 ckpt/fr_t_dsm_s3.pt:10:0.5
echo "[$(date -u +%H:%M) UTC] MORE SEEDS T DONE"
