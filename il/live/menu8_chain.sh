#!/bin/bash
# Menu-8 (extended fixed menu, features8) imitation: all 29,588 seats from bc_v6 (market head surgery), then leader
# fine-tunes (Vadim team 5, M&M&P&Q team 1, DSM team 10).  Benchmarks are run separately once the runtime is verified.
cd /Users/asad.ali/kaggriculture/il
N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 INIT=ckpt/bc_v6.pt \
  nice -n 3 ../.venv/bin/python train8.py ckpt/bc_v8m.pt 1 96 data/union > logs/train_v8m.log 2>&1
echo "[$(date +%H:%M)] all-data: $(grep 'best val loss' logs/train_v8m.log)"
for L in vadim mmpq dsm; do
  N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v8m.pt \
    nice -n 3 ../.venv/bin/python train8.py ckpt/bc_v8m_$L.pt 2 96 data/lead_$L > logs/train_v8m_$L.log 2>&1
  echo "[$(date +%H:%M)] $L: $(grep 'best val loss' logs/train_v8m_$L.log)"
done
echo "[$(date +%H:%M)] menu-8 training done"
