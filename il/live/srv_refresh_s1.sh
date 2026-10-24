#!/bin/bash
# Sep 26 ~12:40 UTC: the GPU is idle after the seed-0 refresh (done 12:30), so a second seed pair (unweighted u / rating-
# weighted w, seed 1) of the same recipe: 2 passes over fresh2 at 3e-4 from fr_t_s0, then the DSM fine-tune (2 x dsm_ft2
# at 2e-4). No Boey fine-tune (its val loss 1.16 vs 0.43 for DSM). LOAD_THREADS 2 to keep the CPU total near the
# operator's ~50-core budget. Benchmarks are queued later (the bench lock is busy for hours).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
RW="RATING_W=/Users/asad.ali/kaggriculture/il/data/rating_w_fresh2.json"
run() {  # $1 tag  $2 extra env (or -)
  local E=""; [ "$2" != "-" ] && E="$2"
  env SEED=1 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=3e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=2 PREFETCH=1 \
    INIT=ckpt/fr_t_s0.pt $E $PY train6.py ckpt/rf_$1_s1.pt 2 256 data/fresh2 > logs/train_rf_$1_s1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] refresh pass done $1 s1: $(grep 'best val loss' logs/train_rf_$1_s1.log)" | tee -a logs/refresh.log
  [ -f ckpt/rf_$1_s1.pt ] && env SEED=1 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=2 PREFETCH=1 \
    INIT=ckpt/rf_$1_s1.pt $PY train6.py ckpt/rf_$1_dsm_s1.pt 2 256 data/dsm_ft2 > logs/train_rf_$1_dsm_s1.log 2>&1
  echo "[$(date -u +%H:%M) UTC] refresh dsm FT done $1 s1: $(grep 'best val loss' logs/train_rf_$1_dsm_s1.log)" | tee -a logs/refresh.log
}
run u - & sleep 60; run w "$RW" & wait
echo "[$(date -u +%H:%M) UTC] REFRESH S1 DONE" | tee -a logs/refresh.log
