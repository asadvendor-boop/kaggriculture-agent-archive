#!/bin/bash
# Growth test on the rented A100 (operator, Sep 25): seeds 1 and 2 for BOTH sizes - same-size v6 (96 wide, 5 blocks)
# vs grown 4M (176 wide, 6 blocks, function-preserving copy of bc_v6) - each one more pass over all 29,588 seats
# (exactly the Mac recipe: 5e-4, 48 teams, chunk 100) then the same Vadim fine-tune (2 epochs, 2e-4).  Seed 0 of both
# runs on the Mac.  fp32 (no AMP) so the numbers stay comparable with the Mac runs; LOAD_THREADS/PREFETCH only change speed.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
run() {   # $1 tag (c = same size, g = grown)  $2 seed  $3 width  $4 blocks  $5 init
  env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=8 PREFETCH=1 INIT=$5 \
    $PY train6.py ckpt/bc_v6$1_all_s$2.pt 1 $3 data/union > logs/train_v6$1_all_s$2.log 2>&1
  echo "[$(date -u +%H:%M) UTC] pass done $1 s$2: $(grep 'best val loss' logs/train_v6$1_all_s$2.log)"
  env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=8 PREFETCH=1 INIT=ckpt/bc_v6$1_all_s$2.pt \
    $PY train6.py ckpt/bc_v6$1_all_vadim_s$2.pt 2 $3 data/vadim > logs/train_v6$1_all_vadim_s$2.log 2>&1
  echo "[$(date -u +%H:%M) UTC] Vadim FT done $1 s$2: $(grep 'best val loss' logs/train_v6$1_all_vadim_s$2.log)"
}
run c 1 96 5 ckpt/bc_v6.pt &
run c 2 96 5 ckpt/bc_v6.pt &
run g 1 176 6 ckpt/grow_v6_176x6.pt &
run g 2 176 6 ckpt/grow_v6_176x6.pt &
wait
echo "[$(date -u +%H:%M) UTC] ALL DONE"
