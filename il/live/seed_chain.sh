#!/bin/bash
# Is v6-Vadim's 322/464 reproducible?  The exact v6-Vadim fine-tune (bc_v6 -> data/vadim, 2 epochs, 2e-4, width 96)
# with training seeds 1 and 2, then the same benchmark (tau 0.5).  Spread across seeds = training noise of our comparisons.
cd /Users/asad.ali/kaggriculture/il
for s in 1 2; do
  SEED=$s BLOCKS=5 N_TEAMS=45 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v6.pt \
    nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6_vadim_s$s.pt 2 96 data/vadim > logs/train_v6_vadim_s$s.log 2>&1
  echo "[$(date +%H:%M)] seed $s: $(grep 'best val loss' logs/train_v6_vadim_s$s.log)"
  (cd .. && LOCKSTEP=6 NW=4 TEAM=5 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5 nice -n 5 .venv/bin/python il/live/ladder.py eval test il/ckpt/bc_v6_vadim_s$s.pt 2>&1 | grep -v Warning | grep -v recorded) &
done
wait
echo "[$(date +%H:%M)] seed test done"
