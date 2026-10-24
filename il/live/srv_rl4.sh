#!/bin/bash
# Sep 26 evening: RL on the 20M imitation model (g20_u_dsm_s0: 440/464, 2400+ 211/219, today's top-10 70-52 = our best
# model vs the top band), same league recipe as RL3 (strong style clones DSM/Vadim/DECEM/UMG/Boey, recorded leader games,
# ladder, live losses, FORK_M, RACE5, self) with GPU rollout inference. 24 iterations or 5 h; evaluations every 6; then
# the 464 worlds and both leader tests for every checkpoint, paired against the 20M itself.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
C=/Users/asad.ali/kaggriculture/il/ckpt
INIT=$C/g20_u_dsm_s0.pt; TEAM=10; LEAD=$C/fr_t_s0.pt
P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
echo "[$(date -u +%H:%M) UTC] RL4 (20M) starts from $INIT"
timeout 18000 env ROLLOUT_DEVICE=cuda NW=${NW_RL:-16} GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=768 EVAL=6 EVAL_N=50 EVAL_LADDER=232 \
  EVAL_OPPS="$P5,FORK_M,NN:$LEAD:10:0.5,NN:$LEAD:0:0.5" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$INIT LOSS_SUBS=56553797,56553911,56559691,56571775 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="Boey|DSM|M & M & P & Q|Fourth Quadrant|DECEM|Unknown Mother-Goose|Azat Akhtyamov|🐚seek inspiration🐚|Majkel1337|THIRD FARM CLUB|吃白饭的大肥鱼" \
  LEAD_CKPT=$LEAD LEAD_TEAMS="10,10,10,5,5,3,3,4,4,0,0" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="LEADERS=4,TOP=4,LADDER=1.5,LOSSES=1,FORK_M=1,$P5=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl4.pt 24 > logs/ppo_rl4.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL4 finished: $(grep -c '^\[eval' logs/ppo_rl4.log) evaluations"
CK="$(ls ckpt/rl4_it*.pt 2>/dev/null | sort -V | tr '\n' ' ')"
bench_gov ladder $(for c in $CK; do echo $c:$TEAM:0.5; done)
for c in $CK; do $PY live/pair_cmp.py logs/ladder_g20_u_dsm_s0_t10_r3_tau0.5_test.json logs/ladder_$(basename $c .pt)_t10_r3_tau0.5_test.json; done
bench_gov top:10 $CK
bench_gov top2:10 $CK
for c in $CK; do
  $PY live/pair_cmp.py logs/top_fresh_g20_u_dsm_s0_t10.json logs/top_fresh_$(basename $c .pt)_t10.json
  $PY live/pair_cmp.py logs/top_fresh2_g20_u_dsm_s0_t10.json logs/top_fresh2_$(basename $c .pt)_t10.json
done
echo "[$(date -u +%H:%M) UTC] RL4 CHAIN DONE"
