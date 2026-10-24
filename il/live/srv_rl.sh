#!/bin/bash
# RL (operator, Sep 25: "fix all the gaps before training then do IL and end with RL"). Starts once the 1.33M and 4M
# DSM-style models (fresh old-menu DSM fine-tunes + gap-fixed menu-8 DSM+DECEM fine-tunes) are benchmarked (or at 21:30
# UTC at the latest), from the best of them on the 464 ladder worlds. Near-greedy PPO against live-style opponents:
# fresh top-12 leader games (TOP), our ladder games (LADDER), our live losses (LOSSES), the copycat proxy FORK_M, HighWool,
# STACK1_RACE5/4, V94 and snapshots of itself; evaluation every 6 iterations on 232 held-out ladder worlds + 100 games each
# vs STACK1_RACE5 and FORK_M. Stops after 48 iterations or at 08:30 UTC (the machine stops 11:27 UTC), then benchmarks
# every saved checkpoint on the full 464 ladder worlds. CPU: 14 RL workers (benchmarks keep the governor's share).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
DL=$(date -u -d "21:30" +%s)
ready() { for k in c_s0 c_s1 g_s0 g_s1; do
            [ -f logs/ladder_fr_${k%_s*}_dsm_s${k#*_s}_t10_r3_tau0.5_test.json ] || return 1
            [ -f logs/g8_${k%_s*}_s${k#*_s}.finished ] || return 1
            [ -f ckpt/g8_${k%_s*}_dd_s${k#*_s}.pt ] && { [ -f logs/ladder_g8_${k%_s*}_dd_s${k#*_s}_t3_r3_tau0.5_test.json ] || return 1; }
          done; return 0; }
until ready || [ $(date -u +%s) -ge $DL ]; do sleep 120; done
echo "[$(date -u +%H:%M) UTC] candidates (464 ladder worlds):"
read INIT TEAM W M < <($PY live/select_best.py 2>> logs/rl.log)
echo "[$(date -u +%H:%M) UTC] RL starts from $INIT (team $TEAM): $W wins, margin $M"
P4=/Users/asad.ali/kaggriculture/package/STACK1_RACE4/main.py; P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
LEFT=$(( $(date -u -d "08:30 tomorrow" +%s) - $(date -u +%s) ))
timeout $LEFT env NW=14 GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=1024 EVAL=6 EVAL_N=100 EVAL_LADDER=232 EVAL_OPPS="$P5,FORK_M" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$INIT LOSS_SUBS=56553797,56553911 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="DSM|Boey|M & M & P & Q|Unknown Mother-Goose|DECEM|Fourth Quadrant|Majkel1337|mtmr_s1|Vadim Vasilenko|吃白饭的大肥鱼|Kaggledew Valley 🏆|THIRD FARM CLUB" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="TOP=6,LADDER=2,LOSSES=1,FORK_M=2,HighWool=1,$P5=1,$P4=0.5,V94=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl1.pt 48 > logs/ppo_rl1.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL finished: $(grep -c '^\[eval' logs/ppo_rl1.log) evaluations; $(grep 'new best' logs/ppo_rl1.log | tail -1)"
grep "^\[eval" logs/ppo_rl1.log | cut -c1-260
for c in $(ls ckpt/rl1_it*.pt 2>/dev/null | sort -V); do bench_gov ladder $c:$TEAM:0.5; done
echo "[$(date -u +%H:%M) UTC] RL CHAIN DONE"
