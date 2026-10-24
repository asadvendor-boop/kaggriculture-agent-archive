#!/bin/bash
# Sep 26 ~13:25 UTC: RL3 continues from its iteration-6 checkpoint (rl3_it6.pt: policy, value net, Adam states) with the
# policy forward of the 20 rollout workers on the GPU (ROLLOUT_DEVICE=cuda; measured 4x faster per worker, same results).
# Same recipe, opponents and KL anchor (REF = rl2_it24); 24 more iterations (= RL3 iterations 7-30) or 5 h; then the 464
# worlds and both leader tests for rl3_it6 and every rl3b checkpoint (rl3b_itN = RL3 iteration 6+N).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
C=/Users/asad.ali/kaggriculture/il/ckpt
INIT=$C/rl3_it6.pt; REF=$C/rl2_it24.pt; TEAM=10; LEAD=$C/fr_t_s0.pt
P5=/Users/asad.ali/kaggriculture/package/STACK1_RACE5/main.py
echo "[$(date -u +%H:%M) UTC] RL3b starts from $INIT on GPU rollout inference, KL anchor $REF"
timeout 18000 env ROLLOUT_DEVICE=cuda NW=${NW_RL:-20} GPW=8 LOCKSTEP=6 MAXTASKS=6 MB=1024 EVAL=6 EVAL_N=50 EVAL_LADDER=232 \
  EVAL_OPPS="$P5,FORK_M,NN:$LEAD:10:0.5,NN:$LEAD:0:0.5" TEAM=$TEAM TAU=0.5 \
  T=0.2 KSHARP=10 LR=5e-6 BETA=0.05 EPOCHS=2 WINB=0 PFSP=1 REF=$REF LOSS_SUBS=56553797,56553911,56559691,56571775 \
  LOSS_DIR=/Users/asad.ali/kaggriculture/data/tapes_nlive TOP_DIRS=data/tapes_top12 \
  TOP_TEAMS="Boey|DSM|M & M & P & Q|Fourth Quadrant|DECEM|Unknown Mother-Goose|Azat Akhtyamov|🐚seek inspiration🐚|Majkel1337|THIRD FARM CLUB|吃白饭的大肥鱼" \
  LEAD_CKPT=$LEAD LEAD_TEAMS="10,10,10,5,5,3,3,4,4,0,0" \
  MIDNIGHT=1 FINALFIX=1 HOMING=1 OPPS="LEADERS=4,TOP=4,LADDER=1.5,LOSSES=1,FORK_M=1,$P5=0.5,SELF=1" \
  $PY ppo_live.py $INIT ckpt/rl3b.pt 24 > logs/ppo_rl3b.log 2>&1
echo "[$(date -u +%H:%M) UTC] RL3b finished: $(grep -c '^\[eval' logs/ppo_rl3b.log) evaluations"
CK="ckpt/rl3_it6.pt $(ls ckpt/rl3b_it*.pt 2>/dev/null | sort -V | tr '\n' ' ')"
bench_gov ladder $(for c in $CK; do echo $c:$TEAM:0.5; done)
for c in $CK; do $PY live/pair_cmp.py logs/ladder_rl2_it24_t10_r3_tau0.5_test.json logs/ladder_$(basename $c .pt)_t10_r3_tau0.5_test.json; done
bench_gov top:10 $CK
bench_gov top2:10 $CK
for c in $CK; do
  $PY live/pair_cmp.py logs/top_fresh_rl2_it24_t10.json logs/top_fresh_$(basename $c .pt)_t10.json
  $PY live/pair_cmp.py logs/top_fresh2_rl2_it24_t10.json logs/top_fresh2_$(basename $c .pt)_t10.json
done
echo "[$(date -u +%H:%M) UTC] RL3b CHAIN DONE"
