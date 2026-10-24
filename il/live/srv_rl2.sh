#!/bin/bash
# RL on the live model's checkpoint (Sep 26 ~03:25 PKT): rl1 from the 4M (420) peaked at it 24 = 433/464 (+13 wins), then
# declined. Same near-greedy PPO recipe and opponents, now from fr_t_dsm_s0 (10M, 429, live as 56559691); evaluations every
# 6 iterations; stops after 30 iterations or at 02:30 UTC; then the full 464-world benchmark of every checkpoint.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
INIT=/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt; TEAM=10
P4=/Users/asad.ali/kaggriculture/package/STACK1_RACE4/main.py; P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
LEFT=$(( $(date -u -d "02:30 tomorrow" +%s) - $(date -u +%s) )); [ $LEFT -gt 43200 ] && LEFT=$(( $(date -u -d "02:30" +%s) - $(date -u +%s) ))
echo "[$(date -u +%H:%M) UTC] RL2 starts from $INIT (team $TEAM), time limit $((LEFT / 60)) min"
timeout $LEFT env NW=14 GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=1024 EVAL=6 EVAL_N=100 EVAL_LADDER=232 EVAL_OPPS="$P5,FORK_M" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$INIT LOSS_SUBS=56553797,56553911,56559691 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="DSM|Boey|M & M & P & Q|Unknown Mother-Goose|DECEM|Fourth Quadrant|Majkel1337|mtmr_s1|Vadim Vasilenko|吃白饭的大肥鱼|Kaggledew Valley 🏆|THIRD FARM CLUB" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="TOP=6,LADDER=2,LOSSES=1,FORK_M=2,HighWool=1,$P5=1,$P4=0.5,V94=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl2.pt 30 > logs/ppo_rl2.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL2 finished: $(grep -c '^\[eval' logs/ppo_rl2.log) evaluations; $(grep 'new best' logs/ppo_rl2.log | tail -1)"
for c in $(ls ckpt/rl2_it*.pt 2>/dev/null | sort -V); do bench_gov ladder $c:$TEAM:0.5; done
echo "[$(date -u +%H:%M) UTC] RL2 CHAIN DONE"
