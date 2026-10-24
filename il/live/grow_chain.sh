#!/bin/bash
# Growth test (operator, Sep 25): does a 4M model grown from our 1.33M model (exact same function at step 0) do better
# or regress?  (1) closed-loop proof: grown v6-Vadim must score 322/464.  (2) SAME DATA: grow bc_v6, then exactly the
# v6-Vadim fine-tune (data/vadim, 2 epochs, 2e-4) -> compare with v6-Vadim 322.  (3) MORE DATA: bc_v6 grown vs not grown,
# each 1 more pass over all 29,588 seats (5e-4) then the same Vadim fine-tune -> grown vs same-size control.
cd /Users/asad.ali/kaggriculture
bench() {  # $1 ckpt, $2 tau
  LOCKSTEP=6 NW=5 TEAM=5 TAU=$2 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau$2 nice -n 5 .venv/bin/python il/live/ladder.py eval test il/ckpt/$1.pt 2>&1 | grep -v Warning | grep -v recorded
}
bench grow_v6vadim_176x6 0.5 &
.venv/bin/python il/grow.py il/ckpt/bc_v6.pt il/ckpt/grow_v6_176x6.pt 176 6 2>&1 | grep -v Warning
cd il
BLOCKS=6 N_TEAMS=45 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/grow_v6_176x6.pt \
  nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6g_vadim.pt 2 176 data/vadim > logs/train_v6g_vadim.log 2>&1
echo "[$(date +%H:%M)] (2) grown + Vadim FT: $(grep 'best val loss' logs/train_v6g_vadim.log)  (v6-Vadim: 0.4803)"
cd ..
wait
for t in 0.5 0.45 0.55; do bench bc_v6g_vadim $t; done &
cd il
BLOCKS=5 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 INIT=ckpt/bc_v6.pt \
  nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6c_all.pt 1 96 data/union > logs/train_v6c_all.log 2>&1
echo "[$(date +%H:%M)] (3) control +1 pass: $(grep 'best val loss' logs/train_v6c_all.log)"
BLOCKS=6 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 INIT=ckpt/grow_v6_176x6.pt \
  nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6g_all.pt 1 176 data/union > logs/train_v6g_all.log 2>&1
echo "[$(date +%H:%M)] (3) grown +1 pass: $(grep 'best val loss' logs/train_v6g_all.log)"
for m in c:96:5 g:176:6; do IFS=: read s w b <<< "$m"
  BLOCKS=$b N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v6${s}_all.pt \
    nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6${s}_all_vadim.pt 2 $w data/vadim > logs/train_v6${s}_all_vadim.log 2>&1
  echo "[$(date +%H:%M)] (3) $s + Vadim FT: $(grep 'best val loss' logs/train_v6${s}_all_vadim.log)"
done
cd ..
wait
for s in c g; do for t in 0.5 0.45 0.55; do bench bc_v6${s}_all_vadim $t; done; done
echo "[$(date +%H:%M)] growth test done"
