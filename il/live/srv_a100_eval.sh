#!/bin/bash
# Sep 27 night, A100 box (~46-core quota): tests models on our rulers WITH the SE guard, as they would ship. Leader tests
# (top:10 + top2:10) for every listed model first, then the 464 ladder worlds for the TOP_N best of them by leader-test
# wins (old leaders + today's top-10), all paired vs v13_p3 (NEURAL_20MW8_SE, the pick) and vs NEURAL_20MW5_SE (live).
# Then (not for the H100 RL checkpoints) the live-loss replay: every recorded live loss of both active submissions.
# Output: /root/a100_bench.log.   Usage: srv_a100_eval.sh <model> [<model> ...]   (ckpt/<model>.pt)   env TOP_N (3)
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
source live/srv_cpu.sh; export CPU_BUDGET=${CPU_BUDGET_E:-40} BENCH_CAP=${BENCH_CAP_E:-36} SE_GUARD=1   # quota 46: keep 4-6 free
pair() {   # pair <label> <base json> <cand json>
  $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-240
}
declare -A W
for m in "$@"; do
  [ -f ckpt/$m.pt ] || { echo "== $m: no checkpoint" >> $L; continue; }
  if ! grep -qx "== $m leader tests" $L 2>/dev/null; then
    TAGX=_se bench_gov top:10 ckpt/$m.pt > logs/a100_${m}_top.log 2>&1
    TAGX=_se bench_gov top2:10 ckpt/$m.pt > logs/a100_${m}_top2.log 2>&1
    { echo "== $m leader tests"
      for d in fresh fresh2; do
        pair "vs v13_p3  " logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_${m}_t10_se.json
        pair "vs 20MW5_SE" logs/top_${d}_g20_w5_dsm_s0_t10_se.json logs/top_${d}_${m}_t10_se.json
      done; } >> $L
  fi
  W[$m]=$($PY -c "
import json
w = 0
for d in ('fresh', 'fresh2'):
    try: w += sum(r['margin'] > 0 for r in json.load(open('logs/top_%s_${m}_t10_se.json' % d)))
    except Exception: pass
print(w)")
done
# live-loss replay (Sep 27 night): every recorded live loss of both active submissions, candidate vs the v13_p3 baseline
# (the baseline is re-run whenever the loss set changed); the operator's "counters to our live weaknesses" test
liveloss() {   # liveloss <model>
  local hs; hs=$($PY live/srv_liveloss_worlds.py | awk '{print $2}')
  if [ "$(cat logs/liveloss_base.hash 2>/dev/null)" != "$hs" ]; then
    FRESH=1 DAY=livelosses N=300 NW=${LL_NW:-12} TEAM=10 TAU=0.5 SE_GUARD=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_se \
      $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/a100_ll_base.log 2>&1 && echo $hs > logs/liveloss_base.hash
  fi
  FRESH=1 DAY=livelosses N=300 NW=${LL_NW:-12} TEAM=10 TAU=0.5 SE_GUARD=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_se \
    $PY live/top_bench.py ckpt/$1.pt > logs/a100_${1}_ll.log 2>&1
  { echo "== $1 live losses ($(cat logs/liveloss_base.hash))"; grep -h -E " n [0-9]+ " logs/a100_${1}_ll.log | cut -c1-150
    pair "vs v13_p3   live losses" logs/top_livelosses_ilsp_v13_p3_t10_se.json logs/top_livelosses_${1}_t10_se.json; } >> $L
}
for m in "$@"; do case $m in rlc_*) ;; *) [ -f ckpt/$m.pt ] && liveloss $m ;; esac; done
for m in $(for k in "${!W[@]}"; do echo "${W[$k]} $k"; done | sort -rn | head -${TOP_N:-3} | awk '{print $2}'); do
  grep -qx "== $m 464" $L 2>/dev/null && continue
  SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$m.pt:10:0.5 > logs/a100_${m}_464.log 2>&1
  { echo "== $m 464"; grep -E " n 464 " logs/a100_${m}_464.log | cut -c1-200
    pair "vs v13_p3   464" logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_${m}_t10_r3_tau0.5_se_test.json
    pair "vs 20MW5_SE 464" logs/ladder_g20_w5_dsm_s0_t10_r3_tau0.5_se_test.json logs/ladder_${m}_t10_r3_tau0.5_se_test.json; } >> $L
done
echo "[$(date -u +%H:%M) UTC] a100 eval done: $*" >> $L
