#!/bin/bash
# Sep 26 evening: the SE guard (the only rule that survived every ruler) on the 20M imitation model (our best vs the top
# band), for the package decision: 464 worlds + both leader tests, paired vs the 20M without the rule.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M=ckpt/g20_u_dsm_s0.pt
SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder $M:10:0.5
$PY live/pair_cmp.py logs/ladder_g20_u_dsm_s0_t10_r3_tau0.5_test.json logs/ladder_g20_u_dsm_s0_t10_r3_tau0.5_se_test.json
SE_GUARD=1 TAGX=_se bench_gov top:10 $M
SE_GUARD=1 TAGX=_se bench_gov top2:10 $M
$PY live/pair_cmp.py logs/top_fresh_g20_u_dsm_s0_t10.json logs/top_fresh_g20_u_dsm_s0_t10_se.json
$PY live/pair_cmp.py logs/top_fresh2_g20_u_dsm_s0_t10.json logs/top_fresh2_g20_u_dsm_s0_t10_se.json
echo "[$(date -u +%H:%M) UTC] G20 SE DONE"
