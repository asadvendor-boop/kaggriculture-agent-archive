#!/bin/bash
# Sep 26 ~20:20 UTC: RL5 = the RL4 recipe (league: DSM/leader-style clones, pinned recorded leader games, ladder, live
# losses, FORK_M, RACE5, self; GPU rollout inference) started from the WEIGHTED 20M (g20_w_dsm_s0 + SE = our upload
# candidate: 448/464, 2400+ 208, old leaders 93, top-10 67). RL4 (from the unweighted 20M) at it12 + SE reached
# 440 / 207 / 92 / 67. Waits for RL4's PPO process to exit. After RL5: SE benchmarks of it12 and it24 (464 + both leader
# tests) paired vs g20_w + SE.
exec 8>/tmp/srv_rl5.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
while pgrep -f "[p]po_live.py .*ckpt/rl4.pt" >/dev/null; do sleep 60; done; sleep 30
C=/Users/asad.ali/kaggriculture/il/ckpt
INIT=$C/g20_w_dsm_s0.pt; TEAM=10; LEAD=$C/fr_t_s0.pt
P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
echo "[$(date -u +%H:%M) UTC] RL5 (weighted 20M) starts from $INIT"
timeout 18000 env ROLLOUT_DEVICE=cuda NW=16 GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=768 EVAL=6 EVAL_N=50 EVAL_LADDER=232 \
  EVAL_OPPS="$P5,FORK_M,NN:$LEAD:10:0.5,NN:$LEAD:0:0.5" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$INIT LOSS_SUBS=56553797,56553911,56559691,56571775 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="Boey|DSM|M & M & P & Q|Fourth Quadrant|DECEM|Unknown Mother-Goose|Azat Akhtyamov|🐚seek inspiration🐚|Majkel1337|THIRD FARM CLUB|吃白饭的大肥鱼" \
  LEAD_CKPT=$LEAD LEAD_TEAMS="10,10,10,5,5,3,3,4,4,0,0" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="LEADERS=4,TOP=4,LADDER=1.5,LOSSES=1,FORK_M=1,$P5=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl5.pt 24 > logs/ppo_rl5.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL5 finished: $(grep -c '^\[eval' logs/ppo_rl5.log) evaluations"
for M in rl5_it12 rl5_it24; do
  [ -f ckpt/$M.pt ] || continue
  SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$M.pt:10:0.5
  $PY live/pair_cmp.py logs/ladder_g20_w_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${M}_t10_r3_tau0.5_se_test.json
  SE_GUARD=1 TAGX=_se bench_gov top:10 ckpt/$M.pt
  SE_GUARD=1 TAGX=_se bench_gov top2:10 ckpt/$M.pt
  for d in fresh fresh2; do $PY live/pair_cmp.py logs/top_${d}_g20_w_dsm_s0_t10_se.json logs/top_${d}_${M}_t10_se.json; done
done
echo "[$(date -u +%H:%M) UTC] RL5 CHAIN DONE"
