#!/bin/bash
# After menu8_chain: menu-8 fine-tunes for the leaders the old menu represented worst, then the full 464 ladder benchmark
# at tau 0.5 for every menu-8 fine-tune (team id of the imitated leader).
cd /Users/asad.ali/kaggriculture/il
until grep -q "menu-8 training done" logs/menu8_chain.log 2>/dev/null; do sleep 60; done
for L in tfc mtmr kdew cbf majkel snorlax boey fq; do
  N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v8m.pt \
    nice -n 3 ../.venv/bin/python train8.py ckpt/bc_v8m_$L.pt 2 96 data/lead8_$L > logs/train_v8m_$L.log 2>&1
  echo "[$(date +%H:%M)] $L: $(grep 'best val loss' logs/train_v8m_$L.log)"
done
echo "[$(date +%H:%M)] fine-tunes done"
