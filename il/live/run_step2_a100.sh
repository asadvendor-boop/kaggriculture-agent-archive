#!/bin/bash
# Growth test step 2 with several seeds (operator, Sep 25): "same data, bigger model" - the EXACT v6-Vadim fine-tune
# (bc_v6 -> data/vadim, 2 epochs, 2e-4, 45 teams) from the same-size bc_v6 (96/5) and from its function-preserving
# 4M copy grow_v6_176x6 (176/6), seeds 1-4 each, 8 runs in parallel on the A100 (fp32, as on the Mac).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
run() {   # $1 tag  $2 seed  $3 width  $4 blocks  $5 init
  env SEED=$2 BLOCKS=$4 N_TEAMS=45 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 INIT=$5 \
    $PY train6.py ckpt/s2_$1_s$2.pt 2 $3 data/vadim > logs/s2_$1_s$2.log 2>&1
  echo "[$(date -u +%H:%M) UTC] $1 s$2: $(grep 'best val loss' logs/s2_$1_s$2.log)"
}
for s in 1 2 3 4; do
  run small $s 96 5 ckpt/bc_v6.pt &
  run grown $s 176 6 ckpt/grow_v6_176x6.pt &
done
wait
echo "[$(date -u +%H:%M) UTC] STEP2 DONE"
