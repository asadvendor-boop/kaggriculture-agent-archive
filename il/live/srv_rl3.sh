#!/bin/bash
# Sep 26, operator-approved actions 2 + 4: LEAGUE RL against leader-style opponents, continued from the best RL model
# (rl2_it24 = 434/464, live as 56571775).  Reactive leader clones: the fresh multi-style 10M imitation model (fr_t_s0,
# before the DSM fine-tune) playing as a drawn leader style, weighted toward TODAY's top 10 (operator screenshot Sep 26:
# Boey 3116, DSM 3028, M&M&P&Q 3017, Fourth Quadrant 2974, DECEM 2968, UMG 2936, Azat 2929, seek inspiration 2910,
# Majkel 2901, THIRD FARM CLUB 2900) and the matchups we lose on the held-out leader benchmark (Boey 25%, DSM 25%,
# THIRD FARM CLUB 18%, 吃白饭的大肥鱼 8%).  Style ids: Boey 0, M&M&P&Q 1, DECEM 3, UMG 4, 大肥鱼 6, TFC 7, DSM 10,
# Majkel 11, Fourth Quadrant 45, Azat 46 (seek inspiration has no style id: recorded games only).  Recorded leader games
# (TOP) of today's top 10 + 大肥鱼; live losses of all four neural submissions; FORK_M / RACE5 / SELF keep the ladder mix.
# Evaluations every 6 iterations (incl. the Boey, 大肥鱼 and TFC clones); stops after 30 iterations or $1 hours
# (default 7); then the 464-world benchmark of every checkpoint.
# RESTARTED Sep 26 ~11:50 UTC: the style clones of 吃白饭的大肥鱼 / THIRD FARM CLUB / M&M&P&Q / Fourth Quadrant /
# Majkel / Azat play far below the real teams (idle units, few seeds; lost 33-122k to RACE5 while the DSM, Vadim and
# DECEM clones beat it), so the reactive pool keeps only DSM x3, Vadim x2, DECEM x2, UMG x2, Boey x2; the weak styles
# stay covered by their REAL recorded games (TOP).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
H=${1:-7}
C=/Users/asad.ali/kaggriculture/il/ckpt
INIT=$C/rl2_it24.pt; TEAM=10; LEAD=$C/fr_t_s0.pt
P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
echo "[$(date -u +%H:%M) UTC] RL3 (league) starts from $INIT (team $TEAM), leader clones $LEAD, time limit ${H} h"
timeout $((H * 3600)) env NW=${NW_RL:-12} GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=1024 EVAL=6 EVAL_N=50 EVAL_LADDER=232 \
  EVAL_OPPS="$P5,FORK_M,NN:$LEAD:10:0.5,NN:$LEAD:0:0.5" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$INIT LOSS_SUBS=56553797,56553911,56559691,56571775 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="Boey|DSM|M & M & P & Q|Fourth Quadrant|DECEM|Unknown Mother-Goose|Azat Akhtyamov|🐚seek inspiration🐚|Majkel1337|THIRD FARM CLUB|吃白饭的大肥鱼" \
  LEAD_CKPT=$LEAD LEAD_TEAMS="10,10,10,5,5,3,3,4,4,0,0" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="LEADERS=4,TOP=4,LADDER=1.5,LOSSES=1,FORK_M=1,$P5=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl3.pt 30 > logs/ppo_rl3.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL3 finished: $(grep -c '^\[eval' logs/ppo_rl3.log) evaluations; $(grep 'new best' logs/ppo_rl3.log | tail -1)"
for c in $(ls ckpt/rl3_it*.pt 2>/dev/null | sort -V); do bench_gov ladder $c:$TEAM:0.5; done
for c in $(ls ckpt/rl3_it*.pt 2>/dev/null | sort -V); do $PY live/pair_cmp.py logs/ladder_rl2_it24_t10_r3_tau0.5_test.json logs/ladder_$(basename $c .pt)_t10_r3_tau0.5_test.json; done
# operator rule (Sep 26): judge mainly on the 2,400+ games and the leader tests -> both leader tests for every checkpoint
bench_gov top:10 ckpt/rl2_it24.pt $(ls ckpt/rl3_it*.pt 2>/dev/null | sort -V)
[ -f live/top_worlds_fresh2_300.json ] && bench_gov top2:10 ckpt/rl2_it24.pt $(ls ckpt/rl3_it*.pt 2>/dev/null | sort -V)
for c in $(ls ckpt/rl3_it*.pt 2>/dev/null | sort -V); do
  $PY live/pair_cmp.py logs/top_fresh_rl2_it24_t10.json logs/top_fresh_$(basename $c .pt)_t10.json
  [ -f logs/top_fresh2_rl2_it24_t10.json ] && $PY live/pair_cmp.py logs/top_fresh2_rl2_it24_t10.json logs/top_fresh2_$(basename $c .pt)_t10.json
done
echo "[$(date -u +%H:%M) UTC] RL3 CHAIN DONE"
