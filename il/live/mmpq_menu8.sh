#!/bin/bash
# Menu effect alone on the M&M&P&Q imitation (old menu: 2/464): bc_v6_mmpq with its trunk and unit heads FROZEN,
# market layers (m1, m2, mk after surgery to menu 8) fine-tuned on M&M&P&Q seats, then the full 464 benchmark (team 1).
cd /Users/asad.ali/kaggriculture/il
FREEZE=1 N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v6_mmpq.pt \
  nice -n 3 ../.venv/bin/python train8.py ckpt/bc_v8f_mmpq.pt 2 96 data/lead_mmpq > logs/train_v8f_mmpq.log 2>&1
echo "[$(date +%H:%M)] $(grep 'best val loss' logs/train_v8f_mmpq.log)"
cd ..
for tau in 0.5 0.6; do
  LOCKSTEP=6 NW=4 TEAM=1 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau$tau nice -n 5 .venv/bin/python il/live/ladder.py eval test il/ckpt/bc_v8f_mmpq.pt 2>&1 | grep -v Warning | grep -v recorded
done
echo "[$(date +%H:%M)] done"
