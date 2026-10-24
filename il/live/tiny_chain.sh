#!/bin/bash
# Capacity test, third point: tiny (w48 b4, 0.60M) on the same 12k subset, 1 epoch from scratch, same Vadim fine-tune, benchmarks.
cd /Users/asad.ali/kaggriculture/il
BLOCKS=4 N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=1e-3 CHUNK=100 VAL_MAX=180 AUX_W=0.1 nice -n 3 ../.venv/bin/python train6.py ckpt/cap_tiny.pt 1 48 data/cap12k > logs/train_cap_tiny.log 2>&1
echo "[$(date +%H:%M)] tiny base: $(grep 'best val loss' logs/train_cap_tiny.log)"
BLOCKS=4 N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/cap_tiny.pt nice -n 3 ../.venv/bin/python train6.py ckpt/cap_tiny_vadim.pt 2 48 data/lead_vadim > logs/train_cap_tiny_vadim.log 2>&1
echo "[$(date +%H:%M)] tiny vadim: $(grep 'best val loss' logs/train_cap_tiny_vadim.log)"
cd ..
for tau in 0.5 0.6; do LOCKSTEP=6 NW=4 TEAM=5 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau$tau nice -n 3 .venv/bin/python il/live/ladder.py eval test il/ckpt/cap_tiny_vadim.pt 2>&1 | grep -v Warning | grep -v recorded; done
echo "[$(date +%H:%M)] tiny chain done"
