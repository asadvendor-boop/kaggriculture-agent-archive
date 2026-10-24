#!/bin/bash
# Resume the capacity test after the Sep 24 19:40 pause: small (1.33M) is DONE (cap_small.pt, cap_small_vadim.pt);
# big (3.96M) and tiny (0.60M) restart FROM SCRATCH (same 12k subset, 1 epoch, same schedule) for a fair comparison;
# then the Vadim fine-tunes and the full 464-game benchmarks for all three at tau 0.5 and 0.6.
cd /Users/asad.ali/kaggriculture/il
for cfg in "big 176 6" "tiny 48 4"; do set -- $cfg
  BLOCKS=$3 N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=1e-3 CHUNK=100 VAL_MAX=180 AUX_W=0.1 nice -n 3 ../.venv/bin/python train6.py ckpt/cap_$1.pt 1 $2 data/cap12k > logs/train_cap_$1.log 2>&1
  echo "[$(date +%H:%M)] $1 base: $(grep 'best val loss' logs/train_cap_$1.log)"
  BLOCKS=$3 N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/cap_$1.pt nice -n 3 ../.venv/bin/python train6.py ckpt/cap_$1_vadim.pt 2 $2 data/lead_vadim > logs/train_cap_$1_vadim.log 2>&1
  echo "[$(date +%H:%M)] $1 vadim: $(grep 'best val loss' logs/train_cap_$1_vadim.log)"
done
cd ..
for m in tiny small big; do for tau in 0.5 0.6; do
  LOCKSTEP=6 NW=6 TEAM=5 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau$tau nice -n 3 .venv/bin/python il/live/ladder.py eval test il/ckpt/cap_${m}_vadim.pt 2>&1 | grep -v Warning | grep -v recorded
done; done
echo "[$(date +%H:%M)] capacity test done"
