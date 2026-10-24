#!/bin/bash
# Sep 26 (operator: "we can arrange a faster GPU if 20M shows progress"; Kaggriculture's limit is 100 MiB per submission,
# a 320x10 model = 19.2M weights = ~72 MB tar.gz): grow the fresh 10M base (fr_t_s0, 256x8) function-preservingly to 320x10,
# train it on the same refresh data as the 10M refresh (fresh2; 3 passes at 4e-4 so the new capacity gets used), then the
# DSM fine-tune (2 x dsm_ft2 at 2e-4).  Compared with the 10M trained on the same data (rf_u_dsm_s0) and the live
# fr_t_dsm_s0 on the 464 worlds (2,400+ first) and both leader tests.  Starts when >= 18 GB of GPU memory is free.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
[ -f ckpt/grow_320x10_frt.pt ] || $PY grow.py ckpt/fr_t_s0.pt ckpt/grow_320x10_frt.pt 320 10
until [ "$(gpu_free)" -ge 18000 ]; do sleep 60; done
echo "[$(date -u +%H:%M) UTC] 20M (320x10) training starts (GPU free $(gpu_free) MiB)"
env SEED=0 BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=4e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
  INIT=ckpt/grow_320x10_frt.pt $PY train6.py ckpt/g20_u_s0.pt 3 320 data/fresh2 > logs/train_g20_u_s0.log 2>&1
echo "[$(date -u +%H:%M) UTC] 20M pass done: $(grep 'best val loss' logs/train_g20_u_s0.log)"
env SEED=0 BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
  INIT=ckpt/g20_u_s0.pt $PY train6.py ckpt/g20_u_dsm_s0.pt 2 320 data/dsm_ft2 > logs/train_g20_u_dsm_s0.log 2>&1
echo "[$(date -u +%H:%M) UTC] 20M DSM FT done: $(grep 'best val loss' logs/train_g20_u_dsm_s0.log)"
bench_gov ladder ckpt/g20_u_dsm_s0.pt:10:0.5
for b in fr_t_dsm_s0 rf_u_dsm_s0; do
  [ -f logs/ladder_${b}_t10_r3_tau0.5_test.json ] && $PY live/pair_cmp.py logs/ladder_${b}_t10_r3_tau0.5_test.json logs/ladder_g20_u_dsm_s0_t10_r3_tau0.5_test.json
done
bench_gov top:10 ckpt/g20_u_dsm_s0.pt
bench_gov top2:10 ckpt/g20_u_dsm_s0.pt
for b in fr_t_dsm_s0 rf_u_dsm_s0 rl2_it24; do
  [ -f logs/top_fresh_${b}_t10.json ] && $PY live/pair_cmp.py logs/top_fresh_${b}_t10.json logs/top_fresh_g20_u_dsm_s0_t10.json
  [ -f logs/top_fresh2_${b}_t10.json ] && $PY live/pair_cmp.py logs/top_fresh2_${b}_t10.json logs/top_fresh2_g20_u_dsm_s0_t10.json
done
echo "[$(date -u +%H:%M) UTC] 20M CHAIN DONE"
