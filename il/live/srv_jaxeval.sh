#!/bin/bash
# Sep 26 night: benchmark the overnight JAX PPO run (agent D, warm start g20_w_dsm_s0) on our usual rulers once it ends:
# waits for kagjax/logs_server/D/overnight/DONE (it holds the final ppo_live .pt path, or SKIPPED); copies that checkpoint
# to ckpt/jax1_final.pt (unique name for the bench outputs) and runs 464 + both leader tests with the SE guard, paired vs
# the weighted 20M + SE (the upload candidate). Gives up at 03:00 UTC if no marker.
exec 8>/tmp/srv_jaxeval.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M=/Users/asad.ali/kaggriculture/kagjax/logs_server/D/overnight/DONE
until [ -f $M ]; do
  [ "$(date -u +%H%M)" -ge 0300 ] && [ "$(date -u +%H%M)" -lt 1200 ] && { echo "[$(date -u +%H:%M) UTC] JAXEVAL SKIPPED (no DONE marker)"; exit 0; }
  sleep 120
done
echo "[$(date -u +%H:%M) UTC] marker: $(cat $M)"
grep -q SKIPPED $M && { echo "[$(date -u +%H:%M) UTC] JAXEVAL SKIPPED (run skipped)"; exit 0; }
P=$(grep -o '/[^ ]*\.pt' $M | tail -1)
[ -f "$P" ] || { echo "[$(date -u +%H:%M) UTC] JAXEVAL SKIPPED (checkpoint not found: $P)"; exit 0; }
cp "$P" ckpt/jax1_final.pt
$PY -c "import torch; c = torch.load('ckpt/jax1_final.pt', map_location='cpu', weights_only=False); print('ckpt keys', sorted(k for k in c if k != 'state')[:12], 'width', c.get('width'), 'feat', c.get('feat'))"
X=jax1_final
SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$X.pt:10:0.5
$PY live/pair_cmp.py logs/ladder_g20_w_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${X}_t10_r3_tau0.5_se_test.json
SE_GUARD=1 TAGX=_se bench_gov top:10 ckpt/$X.pt
SE_GUARD=1 TAGX=_se bench_gov top2:10 ckpt/$X.pt
for d in fresh fresh2; do $PY live/pair_cmp.py logs/top_${d}_g20_w_dsm_s0_t10_se.json logs/top_${d}_${X}_t10_se.json; done
echo "[$(date -u +%H:%M) UTC] JAXEVAL DONE"
