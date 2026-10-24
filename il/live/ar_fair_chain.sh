#!/bin/bash
# Fair test of the order-by-order market head (operator, Sep 24): v6-Vadim's proven network FROZEN (its old heads stay
# exact), the order head trained on ALL 29,588 seats (il/data/union), then fine-tuned on Vadim, then the full 464-world
# benchmark with the fixed decoder (item/qty placeholders as in training).  Waits for the capacity test to free the GPU.
cd /Users/asad.ali/kaggriculture/il
until grep -q "tiny vadim" logs/capacity_resume.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] GPU free, training the order head on all data"
FREEZE=1 N_TEAMS=48 AR_W=1 AR_NZ_W=3 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=1e-3 CHUNK=100 VAL_MAX=180 AUX_W=0.1 INIT=ckpt/bc_v6_vadim.pt \
  nice -n 3 ../.venv/bin/python train7.py ckpt/bc_v8ar_all.pt 1 96 data/union > logs/train_v8ar_all.log 2>&1
echo "[$(date +%H:%M)] all-data: $(grep 'best val loss' logs/train_v8ar_all.log)"
FREEZE=1 N_TEAMS=48 AR_W=1 AR_NZ_W=3 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=3e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v8ar_all.pt \
  nice -n 3 ../.venv/bin/python train7.py ckpt/bc_v8ar_vadim.pt 2 96 data/lead_vadim > logs/train_v8ar_vadim.log 2>&1
echo "[$(date +%H:%M)] vadim: $(grep 'best val loss' logs/train_v8ar_vadim.log)"
cd ..
for tau in 0.5 0.4 0.6; do
  LOCKSTEP=6 NW=4 TEAM=5 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau$tau nice -n 5 .venv/bin/python il/live/ladder.py eval test il/ckpt/bc_v8ar_vadim.pt 2>&1 | grep -v Warning | grep -v recorded
done
echo "[$(date +%H:%M)] fair order-head test done"
