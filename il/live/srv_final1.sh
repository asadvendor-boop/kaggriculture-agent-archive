#!/bin/bash
# Sep 26 ~19:10 UTC: the two package candidates on the same footing (SE guard on, as they would be uploaded).
# (1) the weighted 20M (g20_w: 464 449 vs 440, old leader test 95 vs 78-80, today's top-10 66 vs 70) with the SE guard,
#     paired vs itself without the rule and vs the unweighted 20M with the rule;
# (2) RL4 it12 (RL from the unweighted 20M; eval it12: held-out ladder paired +968) with the SE guard, paired vs its start.
exec 8>/tmp/srv_final1.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
for M in g20_w_dsm_s0 rl4_it12; do
  SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$M.pt:10:0.5
  [ $M = g20_w_dsm_s0 ] && $PY live/pair_cmp.py logs/ladder_g20_w_dsm_s0_t10_r3_tau0.5_test.json logs/ladder_g20_w_dsm_s0_t10_r3_tau0.5_se_test.json
  $PY live/pair_cmp.py logs/ladder_g20_u_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${M}_t10_r3_tau0.5_se_test.json
  SE_GUARD=1 TAGX=_se bench_gov top:10 ckpt/$M.pt
  SE_GUARD=1 TAGX=_se bench_gov top2:10 ckpt/$M.pt
  for d in fresh fresh2; do
    [ $M = g20_w_dsm_s0 ] && $PY live/pair_cmp.py logs/top_${d}_g20_w_dsm_s0_t10.json logs/top_${d}_g20_w_dsm_s0_t10_se.json
    $PY live/pair_cmp.py logs/top_${d}_g20_u_dsm_s0_t10_se.json logs/top_${d}_${M}_t10_se.json
  done
  echo "[$(date -u +%H:%M) UTC] $M + SE DONE"
done
echo "[$(date -u +%H:%M) UTC] FINAL1 DONE"
