#!/bin/bash
# Sep 28 ~14:40 PKT: tonight's batch, third form. run_job plays a job's games in LOCKSTEP (one batched CPU forward per
# step), but srv_tonight(_fast).sh gave every game its own process (4.7 min per game alone, 9+ min with 250 processes on
# the 128 physical cores). Now ~10 games per process (the A100 box ran ~1.1-1.3 min per game this way) and at most ~200
# worker processes at once (operator: keep cores free for the system; ~0.65 GB each -> ~130 GB of 251).
# Same rulers, file names and log (/root/tonight.log). Results produced tonight (newer than /root/tn.out) are reused.
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOCKSTEP=10
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $L; }
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
T0=$(stat -c %Y /root/tn.out)
done_tonight() { [ -f "$1" ] && [ $(stat -c %Y "$1") -gt $T0 ]; }
COMMON="N=300 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1"
declare -A NWD=( [llall]=13 [fresh]=14 [fresh2]=12 )          # 126 / 135 / 119 games -> ~10 per process
declare -A V=( [sy7]="SHEEP_YARN=7" [tom]="TOMATO_SHOPS=2" [cow]="COW_MILK=3" [fg]="FEED_GUARD=1" [se2]="SE_SHOPS=2"
               [str]="STRAW_UP=4" [tomD]="TOMATO_SHOPS=2 TOMATO_FROM_DAY=15 TOMATO_PER_SHOP=4" [can]="SELL_ORDER=canonical"
               [combo]="SHEEP_YARN=7 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4" [sysy]="SHEEP_YARN=7 SE_SHOPS=2"
               [combo2]="SHEEP_YARN=7 SE_SHOPS=2 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4"
               [m10]="MELON_WAVE=8 MELON_DAY=10" )
say "LOCKSTEP batch starts (~10 games per process, <= ~200 workers): $($PY live/liveloss_all.py)"
top() {   # top <day> <ckpt> <tagx> <env...>: one leader-style test (llall / fresh / fresh2) unless done tonight
  local d=$1 CK=$2 TX=$3; shift 3
  done_tonight logs/top_${d}_${CK}_t10$TX.json && return 0
  env SE_GUARD=1 SELL_ORDER=held "$@" DAY=$d NW=${NWD[$d]} $COMMON TAGX=$TX $PY live/top_bench.py ckpt/$CK.pt > logs/tn_${d}$TX.log 2>&1
}
# 0. baselines of both bases on the 126 live losses (13 workers each); smoke tests (8 processes) in the background
top llall ilsp_v13_p3 _soh &
top llall ilsp_v13_newplan _nph &
smoke() {
  local EPS b ck tg
  EPS=$($PY - <<'PY'
import json, gzip, sys
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
W = json.load(open("/Users/asad.ali/kaggriculture/il/live/top_worlds_llall_300.json")); out = []
for w in W:
    d = json.load(gzip.open(w["path"], "rt")); A = d["actions"]; g = kagsim_trace.Game(int(d["seed"]), 720, False)
    for t in range(15 * 24):
        g.step(A[0][t] if isinstance(A[0][t], dict) else dict(PASS), A[1][t] if isinstance(A[1][t], dict) else dict(PASS))
    if sum(x == "YARN_STORE" for x in g.observe(0)["town"]["unlocked_shops"]) >= 2: out.append(str(w["ep"]))
print(",".join(out[:6]))
PY
)
  for b in "ilsp_v13_p3:A" "ilsp_v13_newplan:B"; do
    ck=${b%%:*}; tg=${b##*:}
    EPS=$EPS RULES_ENV="SE_GUARD=1 SELL_ORDER=held SHEEP_YARN=7" $PY live/rule_smoke.py ckpt/$ck.pt 6 llall > logs/tn_smoke_sheep_$tg.log 2>&1 &
    EPS=$EPS RULES_ENV="SE_GUARD=1 SELL_ORDER=held" $PY live/rule_smoke.py ckpt/$ck.pt 6 llall > logs/tn_smoke_nosheep_$tg.log 2>&1 &
    EPS=114547731 RULES_ENV="SE_GUARD=1 SELL_ORDER=held TOMATO_SHOPS=2 COW_MILK=3" $PY live/rule_smoke.py ckpt/$ck.pt 1 llall > logs/tn_smoke_tomato_$tg.log 2>&1 &
    EPS=114547731 RULES_ENV="SE_GUARD=1 SELL_ORDER=held" $PY live/rule_smoke.py ckpt/$ck.pt 1 llall > logs/tn_smoke_notomato_$tg.log 2>&1 &
  done; wait
  for tg in A B; do
    { echo "== base $tg: sheep smoke test on 2+ yarn-store losses $EPS (rule on | off), days 12 / 16 / 20"
      paste -d'|' <(grep -E "== ep|d1[26]:|d20:" logs/tn_smoke_sheep_$tg.log | cut -c1-120) <(grep -E "== ep|d1[26]:|d20:" logs/tn_smoke_nosheep_$tg.log | cut -c1-110)
      echo "== base $tg: tomato/cow smoke test on the Planned Economy loss (rules on | off)"
      paste -d'|' <(grep -E "== ep|d08:|d1[0268]:|d2[24]:" logs/tn_smoke_tomato_$tg.log | cut -c1-130) <(grep -E "== ep|d08:|d1[0268]:|d2[24]:" logs/tn_smoke_notomato_$tg.log | cut -c1-110); } > /tmp/tn_smoke_$tg.txt
    cat /tmp/tn_smoke_$tg.txt >> $L
  done
}
smoke &
# 1. one variant chain: live losses -> fresh -> fresh2 (sequential, <= 14 workers) -> the 464 (47 lockstep jobs, 16 workers) if
#    total wins are not below the base's
run_var() {   # run_var <ckpt> <base tag> <variant tag>
  local CK=$1 BT=$2 t=$3 GO WB WV B=/tmp/tn_blk_$2_$3.txt d
  set -- ${V[$t]}
  for d in llall fresh fresh2; do top $d $CK _${BT}_$t "$@"; done
  while ! done_tonight logs/top_llall_${CK}_t10_$BT.json; do sleep 30; done      # the base's live-loss run (step 0)
  read GO WB WV < <($PY - <<PY
import json
wb = wv = 0
for d in ("llall", "fresh", "fresh2"):
    B = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(f"logs/top_{d}_${CK}_t10_$BT.json"))}
    X = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(f"logs/top_{d}_${CK}_t10_${BT}_$t.json"))}
    for k in B.keys() & X.keys(): wb += B[k] > 0; wv += X[k] > 0
print(int(wv >= wb), wb, wv)
PY
)
  { echo "== $BT + $t (${V[$t]}) paired vs $BT"
    pair "live losses" logs/top_llall_${CK}_t10_$BT.json logs/top_llall_${CK}_t10_${BT}_$t.json
    for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_${CK}_t10_$BT.json logs/top_${d}_${CK}_t10_${BT}_$t.json; done
    echo "  $BT + $t total wins (live losses + both leader tests): $WB -> $WV"; } > $B; cat $B >> $L
  if [ "$GO" = 1 ]; then
    done_tonight logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json || \
      env SE_GUARD=1 SELL_ORDER=held "$@" NW=16 LOCKSTEP=10 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_${BT}_$t \
        $PY live/bench_multi.py ckpt/$CK.pt:10:0.5 > logs/tn_464_${BT}_$t.log 2>&1
    { echo "== $BT + $t 464"; pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } > $B; cat $B >> $L
  fi
}
# 2. all 12 variants on v13_p3 (SOH's network, the better one live) at once: <= 12 x 16 = 192 workers
for t in sy7 se2 can fg sysy str cow tomD combo2 m10 tom combo; do run_var ilsp_v13_p3 soh $t & sleep 2; done
wait; say "SOH BASE DONE"
{ echo "== all live losses, the bases paired vs the old live NEURAL_20MW8_SE (v13_p3 + SE, canonical order)"
  for f in top_llall_ilsp_v13_p3_t10_soh top_llall_ilsp_v13_newplan_t10_nph; do pair "llall" logs/top_llall_ilsp_v13_p3_t10_se.json logs/$f.json; done; } >> $L
# 3. reduced nph base (6 x 16 = 96 workers) + v13_p3 head-to-heads held vs canonical (4 x 16 workers, 5 games each)
for t in sy7 se2 can cow str combo2; do run_var ilsp_v13_newplan nph $t & sleep 2; done
for opp in "$K/agents/ref/V94_2945_main.py:vsV94" "$K/package/NEURAL_20MW8_SE/main.py:vs20mw8"; do
  for so in held canonical; do
    NW=16 SE_GUARD=1 SELL_ORDER=$so $PY live/h2h_rules.py ckpt/ilsp_v13_p3.pt ${opp%%:*} 40 tn_ilsp_v13_p3_${so}_${opp##*:} > logs/tn_h2h_${so}_${opp##*:}.log 2>&1 &
  done
done
wait; say "NPH (reduced) + H2H DONE"
for ot in vsV94 vs20mw8; do
  for so in held canonical; do grep -v -i warn logs/tn_h2h_${so}_$ot.log | tail -1 | sed "s/^/  $so $ot: /" >> $L; done
  $PY live/pair_cmp.py logs/h2h_tn_ilsp_v13_p3_canonical_$ot.json logs/h2h_tn_ilsp_v13_p3_held_$ot.json 2>/dev/null | sed "s/^/  v13_p3 held vs canonical ($ot) /" | cut -c1-200 >> $L
done
say "LOCKSTEP batch done"
