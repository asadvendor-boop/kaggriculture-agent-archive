#!/bin/bash
# Sep 27 night, A100 box (operator: "weighted IL ... with the top leaders' current replays"): once il/data/fresh3 is ready
# (srv_fresh3.sh: fresh2 + the leaders' newest games, rating weights), two rating-weighted imitation runs on the GPU
# (it shares the GPU with the anti-specialist RL run), each followed by the DSM fine-tune (2 epochs, as v13_p3) and the
# rulers with the SE guard (srv_a100_eval.sh, paired vs v13_p3):
#   n1 = v13_p3's pass-stage model (ilsp_v13_p3_p) + 1 pass over fresh3 at 1e-4          (~40 min)
#   n3 = v13_p3's exact recipe on the new data: g20_w5_s0 + 3 passes over fresh3 at 1.5e-4 (~2 h)
# N_TEAMS=50: the team vocabulary gained the 2 teams new in the top 10 (srv_fresh3.sh); 48-team inits are expanded.
# Progress: /root/a100_il.log.
exec 8>/tmp/srv_a100_il.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; LOG=/root/a100_il.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="BLOCKS=10 N_TEAMS=50 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1"
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
until grep -q "FRESH3 READY" /root/fresh3.log 2>/dev/null; do sleep 60; done
say "fresh3 ready: $(ls data/fresh3 | wc -l) seats"
run() {   # run <name> <init> <passes> <lr>
  local n=$1 log=logs/ilsp_$1.log
  say "$n: $3 pass(es) of fresh3 from $2 lr $4"
  env SEED=0 MAX_LR=$4 RATING_W=$K/il/data/rating_w_fresh3.json INIT=$2 $C $PY train6.py ckpt/ilsp_${n}_p.pt $3 320 data/fresh3 >> $log 2>&1
  say "$n pass stage: $(grep 'best val loss' $log | tail -1)"
  env SEED=0 MAX_LR=2e-4 INIT=ckpt/ilsp_${n}_p.pt $C $PY train6.py ckpt/ilsp_${n}.pt 2 320 data/dsm_ft2 >> $log 2>&1
  say "$n DONE: $(grep 'best val loss' $log | tail -1)"
  setsid -f bash live/srv_a100_eval.sh ilsp_$n > /dev/null 2>&1 < /dev/null
}
run n1 ckpt/ilsp_v13_p3_p.pt 1 1e-4
run n3 ckpt/g20_w5_s0.pt 3 1.5e-4
say "IL runs done"
