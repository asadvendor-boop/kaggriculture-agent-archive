#!/bin/bash
# Sep 27 ~01:55 UTC: the 5-pass weighted 20M (no SE: 446 / 2400+ 209 / old leaders 98 / today's top-10 73 = the best
# leader-test win counts so far) with the SE guard, as a package candidate: 464 + both leader tests, paired vs the
# weighted 20M + SE (NEURAL_20MW_SE) and vs itself without the rule. Runs after the JAX-run benchmarks (bench lock).
exec 8>/tmp/srv_g20w5se.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M=g20_w5_dsm_s0
SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$M.pt:10:0.5
for b in g20_w_dsm_s0_t10_r3_tau0.5_se g20_w5_dsm_s0_t10_r3_tau0.5; do $PY live/pair_cmp.py logs/ladder_${b}_test.json logs/ladder_${M}_t10_r3_tau0.5_se_test.json; done
SE_GUARD=1 TAGX=_se bench_gov top:10 ckpt/$M.pt
SE_GUARD=1 TAGX=_se bench_gov top2:10 ckpt/$M.pt
for d in fresh fresh2; do
  $PY live/pair_cmp.py logs/top_${d}_g20_w_dsm_s0_t10_se.json logs/top_${d}_${M}_t10_se.json
  $PY live/pair_cmp.py logs/top_${d}_${M}_t10.json logs/top_${d}_${M}_t10_se.json
done
echo "[$(date -u +%H:%M) UTC] G20W5 SE DONE"
