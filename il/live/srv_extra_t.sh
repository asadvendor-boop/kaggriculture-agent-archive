#!/bin/bash
# Idle GPU while RL2 runs (Sep 26 ~04:35 PKT; the machine runs until RL2 ends anyway): (a) does MORE data help? 10M on
# data/fresh7 (7 days, 11,418 seats; 3 passes = about the same steps) vs data/fresh (4 days, 4 passes); (b) one more 10M
# seed on data/fresh (seed 4). Both + the same DSM fine-tune, then 464 ladder benchmarks (team 10), CPU governor.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
run() {  # $1 name  $2 seed  $3 data dir  $4 passes
  env SEED=$2 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/grow_v6_256x8.pt \
    $PY train6.py ckpt/$1.pt $4 256 $3 > logs/train_$1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] pass done $1: $(grep 'best val loss' logs/train_$1.log)"
  env SEED=$2 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/$1.pt \
    $PY train6.py ckpt/$1_dsm.pt 2 256 data/dsm_ft > logs/train_$1_dsm.log 2>&1
  echo "[$(date -u +%H:%M) UTC] DSM FT done $1: $(grep 'best val loss' logs/train_$1_dsm.log)"
}
run fr7_t_s0 0 data/fresh7 3 & sleep 60; run fr_t_s4 4 data/fresh 4 & wait
bench_gov ladder ckpt/fr7_t_s0_dsm.pt:10:0.5 ckpt/fr_t_s4_dsm.pt:10:0.5
echo "[$(date -u +%H:%M) UTC] EXTRA T SEEDS DONE"
