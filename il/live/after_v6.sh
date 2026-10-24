#!/bin/bash
# Chain after v6 imitation: Vadim-only fine-tune, then held-out ladder benchmarks (rules on, team 5, tau 0.6).
cd /Users/asad.ali/kaggriculture
until grep -q "best val loss" il/logs/train_v6.log; do sleep 30; done
echo "[$(date +%H:%M)] v6 done: $(grep 'best val loss' il/logs/train_v6.log)"
cd il && N_TEAMS=45 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v6.pt \
  nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6_vadim.pt 2 96 data/vadim > logs/train_v6_vadim.log 2>&1
cd ..
echo "[$(date +%H:%M)] vadim fine-tune done: $(grep 'best val loss' il/logs/train_v6_vadim.log)"
for ck in il/ckpt/bc_v6.pt il/ckpt/bc_v6_vadim.pt; do
  LOCKSTEP=6 NW=4 TEAM=5 TAU=0.6 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3 nice -n 8 .venv/bin/python il/live/ladder.py eval test $ck 2>&1 | grep -v Warning | grep -v recorded
done
echo "[$(date +%H:%M)] chain done"
