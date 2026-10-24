#!/bin/bash
# Sep 26 ~16:10 UTC: the two best findings of the day combined - 20M capacity (g20_u_dsm_s0: 440/464, 2400+ 211, today's
# top-10 70-52) and rating-weighted imitation (10M: rf_w 442 vs rf_u 437, top-10 69 vs 63) -> g20_w: the same grown 320x10
# start, 3 passes over fresh2 at 4e-4 WITH rating weights, then the DSM fine-tune. Then 464 + both leader tests.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until [ "$(gpu_free)" -ge 16000 ]; do sleep 60; done
echo "[$(date -u +%H:%M) UTC] 20M weighted training starts (GPU free $(gpu_free) MiB)"
env SEED=0 BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=4e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
  RATING_W=/Users/asad.ali/kaggriculture/il/data/rating_w_fresh2.json INIT=ckpt/grow_320x10_frt.pt \
  $PY train6.py ckpt/g20_w_s0.pt 3 320 data/fresh2 > logs/train_g20_w_s0.log 2>&1
echo "[$(date -u +%H:%M) UTC] 20M weighted pass done: $(grep 'best val loss' logs/train_g20_w_s0.log)"
env SEED=0 BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
  INIT=ckpt/g20_w_s0.pt $PY train6.py ckpt/g20_w_dsm_s0.pt 2 320 data/dsm_ft2 > logs/train_g20_w_dsm_s0.log 2>&1
echo "[$(date -u +%H:%M) UTC] 20M weighted DSM FT done: $(grep 'best val loss' logs/train_g20_w_dsm_s0.log)"
bench_gov ladder ckpt/g20_w_dsm_s0.pt:10:0.5
for b in g20_u_dsm_s0 rf_w_dsm_s0; do $PY live/pair_cmp.py logs/ladder_${b}_t10_r3_tau0.5_test.json logs/ladder_g20_w_dsm_s0_t10_r3_tau0.5_test.json; done
bench_gov top:10 ckpt/g20_w_dsm_s0.pt
bench_gov top2:10 ckpt/g20_w_dsm_s0.pt
for b in g20_u_dsm_s0 rf_w_dsm_s0; do
  $PY live/pair_cmp.py logs/top_fresh_${b}_t10.json logs/top_fresh_g20_w_dsm_s0_t10.json
  $PY live/pair_cmp.py logs/top_fresh2_${b}_t10.json logs/top_fresh2_g20_w_dsm_s0_t10.json
done
echo "[$(date -u +%H:%M) UTC] 20M WEIGHTED CHAIN DONE"
