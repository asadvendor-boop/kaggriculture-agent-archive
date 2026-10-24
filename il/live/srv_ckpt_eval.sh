#!/bin/bash
# Sprint checkpoint tester (Sep 27): watches the JAX runs' exported checkpoints (<run>/pt/*.pt) and tests the NEWEST
# untested one of each run (skipping older ones so it keeps pace) on our rulers with the CPU cores while the GPUs keep training: the 464 ladder worlds (2400+ bucket) and both
# leader tests, WITHOUT the SE guard (the RL runs train without it), paired against (1) the 10M warm start (its start)
# and (2) NEURAL_20MW5_SE (the live best, with SE: 450 / 2400+ 212 / old leaders 100 / top-10 73).
# Usage: srv_ckpt_eval.sh A:<K>/kagjax/runs/sprint_A B:<K>/kagjax/runs/sprint_B      env CPU_BUDGET (110), BENCH_CAP (100)
exec 7>/tmp/srv_ckpt_eval.lock; flock -n 7 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
export CPU_BUDGET=${CPU_BUDGET:-110} BENCH_CAP=${BENCH_CAP:-100}
source live/srv_cpu.sh
SEEN=/tmp/srv_ckpt_eval.seen; touch $SEEN
summ() {  # one line per ruler: wins and paired margin vs the references
  local x=$1
  echo "== $x"
  $PY live/pair_cmp.py logs/ladder_rf_w_dsm_s0_t10_r3_tau0.5_test.json logs/ladder_${x}_t10_r3_tau0.5_test.json | sed 's/^/  vs 10M start  /'
  $PY live/pair_cmp.py logs/ladder_g20_w5_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${x}_t10_r3_tau0.5_test.json | sed 's/^/  vs 20MW5_SE   /'
  for d in fresh fresh2; do
    $PY live/pair_cmp.py logs/top_${d}_rf_w_dsm_s0_t10.json logs/top_${d}_${x}_t10.json | sed "s/^/  vs 10M start  /"
    $PY live/pair_cmp.py logs/top_${d}_g20_w5_dsm_s0_t10_se.json logs/top_${d}_${x}_t10.json | sed "s/^/  vs 20MW5_SE   /"
  done
}
while true; do
  for spec in "$@"; do
    lab=${spec%%:*}; dir=${spec#*:}
    for f in $(ls -t $dir/pt/*.pt 2>/dev/null | head -1); do   # newest export only (keeps pace; older ones skipped)
      grep -qx "$f" $SEEN && continue
      sleep 20                                            # let the exporter finish writing
      x=${lab}_$(basename $f .pt); cp $f ckpt/$x.pt; echo "$f" >> $SEEN
      echo "[$(date -u +%H:%M) UTC] testing $x"
      MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5 bench_gov ladder ckpt/$x.pt:10:0.5 > logs/eval_${x}_ladder.log 2>&1
      bench_gov top:10 ckpt/$x.pt > logs/eval_${x}_top.log 2>&1
      bench_gov top2:10 ckpt/$x.pt > logs/eval_${x}_top2.log 2>&1
      grep -h -E " n (464|138|122) " logs/eval_${x}_*.log | cut -c1-200
      summ $x
      echo "[$(date -u +%H:%M) UTC] done $x"
    done
  done
  sleep 60
done
