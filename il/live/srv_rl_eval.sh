#!/bin/bash
# Sep 27 ~21:15 PKT: tests the GPU RL run with the exact CPU recipe (kagjax/runs/cpurec/s<seed>, warm start v13_p3, trained
# WITH the SE guard) on our rulers WITH the SE guard, as it would ship: both leader tests first, then the 464 ladder worlds;
# paired vs its start (ilsp_v13_p3 = NEURAL_20MW8_SE) and vs NEURAL_20MW5_SE (the live best). Output -> /root/il_bench.log.
#   srv_rl_eval.sh <seed> [<seed> ...]      the newest .pt export of each listed seed, in the given order
exec 7>/tmp/srv_rl_eval.lock; flock -n 7 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
source live/srv_cpu.sh; export CPU_BUDGET=110 BENCH_CAP=100 SE_GUARD=1
for s in "$@"; do
  f=$(ls -t $K/kagjax/runs/cpurec/s$s/pt/*.pt 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "== RL s$s: no .pt export yet" >> /root/il_bench.log; continue; }
  x=rlc_s${s}_$(basename $f .pt); cp $f ckpt/$x.pt
  echo "[$(date -u +%H:%M) UTC] RL test $x (from $f)" >> /root/il_bench.log
  TAGX=_se bench_gov top:10 ckpt/$x.pt > logs/ilsp_bench_${x}_top.log 2>&1
  TAGX=_se bench_gov top2:10 ckpt/$x.pt > logs/ilsp_bench_${x}_top2.log 2>&1
  { echo "== RL $x leader tests"
    for d in fresh fresh2; do
      $PY live/pair_cmp.py logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_${x}_t10_se.json | sed 's/^/  vs its start v13_p3 /' | cut -c1-210
      $PY live/pair_cmp.py logs/top_${d}_g20_w5_dsm_s0_t10_se.json logs/top_${d}_${x}_t10_se.json | sed 's/^/  vs 20MW5_SE        /' | cut -c1-210
    done; } >> /root/il_bench.log
  SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$x.pt:10:0.5 > logs/ilsp_bench_${x}_464.log 2>&1
  { echo "== RL $x 464"; grep -E " n 464 " logs/ilsp_bench_${x}_464.log | cut -c1-200
    $PY live/pair_cmp.py logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_${x}_t10_r3_tau0.5_se_test.json | sed 's/^/  vs its start v13_p3 464 /' | cut -c1-260
    $PY live/pair_cmp.py logs/ladder_g20_w5_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${x}_t10_r3_tau0.5_se_test.json | sed 's/^/  vs 20MW5_SE 464 /' | cut -c1-260; } >> /root/il_bench.log
done
echo "[$(date -u +%H:%M) UTC] RL tests done" >> /root/il_bench.log
