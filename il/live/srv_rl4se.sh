#!/bin/bash
# Sep 26 ~20:20 UTC: RL4's last checkpoint with the SE guard on the 464 + both leader tests (it12 + SE is done), paired
# vs the RL start (g20_u + SE) and the upload candidate (g20_w + SE). Replaces RL4's own no-SE bench block.
exec 8>/tmp/srv_rl4se.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
while pgrep -f "[p]po_live.py .*ckpt/rl4.pt" >/dev/null; do sleep 60; done
M=rl4_it24
SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$M.pt:10:0.5
for b in g20_u_dsm_s0 g20_w_dsm_s0; do $PY live/pair_cmp.py logs/ladder_${b}_t10_r3_tau0.5_se_test.json logs/ladder_${M}_t10_r3_tau0.5_se_test.json; done
SE_GUARD=1 TAGX=_se bench_gov top:10 ckpt/$M.pt
SE_GUARD=1 TAGX=_se bench_gov top2:10 ckpt/$M.pt
for b in g20_u_dsm_s0 g20_w_dsm_s0; do for d in fresh fresh2; do $PY live/pair_cmp.py logs/top_${d}_${b}_t10_se.json logs/top_${d}_${M}_t10_se.json; done; done
echo "[$(date -u +%H:%M) UTC] RL4 it24 + SE DONE"
