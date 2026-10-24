#!/bin/bash
# Sep 28 ~14:35 PKT: faster srv_tonight.sh. Measured on this box (EPYC 7742, quota ~246 cores, 251 GB): 4.7 min per
# single-thread game, ~0.6 GB per worker -> the sequential batch (one stage of <= 200 games at a time) would take ~9.5 h.
# Same rulers, file names and log (/root/tonight.log), but: TWO variant chains run concurrently (each <= 135 workers,
# explicit worker counts, no governor), smoke tests in the background, the whole soh base (v13_p3 = SOH's network, better
# live) first incl. m10 (a second melon wave planted days 10-11, W11), then a reduced nph base, then v13_p3 head-to-heads.
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $L; }
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
COMMON="N=300 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1"
declare -A V=( [sy7]="SHEEP_YARN=7" [tom]="TOMATO_SHOPS=2" [cow]="COW_MILK=3" [fg]="FEED_GUARD=1" [se2]="SE_SHOPS=2"
               [str]="STRAW_UP=4" [tomD]="TOMATO_SHOPS=2 TOMATO_FROM_DAY=15 TOMATO_PER_SHOP=4" [can]="SELL_ORDER=canonical"
               [combo]="SHEEP_YARN=7 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4" [sysy]="SHEEP_YARN=7 SE_SHOPS=2"
               [combo2]="SHEEP_YARN=7 SE_SHOPS=2 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4"
               [m10]="MELON_WAVE=8 MELON_DAY=10" )
say "FAST batch starts (2 chains): $($PY live/liveloss_all.py)"
# 0. the two bases on all live losses, concurrently (v13_p3 + SE canonical = the old live script: done 09:20 UTC)
env SE_GUARD=1 SELL_ORDER=held DAY=llall NW=126 $COMMON TAGX=_soh $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/tn_ll_soh.log 2>&1 &
env SE_GUARD=1 SELL_ORDER=held DAY=llall NW=126 $COMMON TAGX=_nph $PY live/top_bench.py ckpt/ilsp_v13_newplan.pt > logs/tn_ll_nph.log 2>&1 &
wait
{ echo "== all live losses, paired vs the old live NEURAL_20MW8_SE (v13_p3 + SE, canonical order)"
  for f in top_llall_ilsp_v13_p3_t10_soh top_llall_ilsp_v13_newplan_t10_nph; do pair "llall" logs/top_llall_ilsp_v13_p3_t10_se.json logs/$f.json; done; } >> $L
# 1. smoke tests in the background (8 processes): sheep on the 2+ yarn-store losses, tomatoes / cows on Planned Economy
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
# 2. one variant: live losses (126) -> leader tests fresh (135) / fresh2 (119) -> the 464 (78 lockstep jobs) if not worse
run_var() {   # run_var <ckpt> <base tag> <variant tag>
  local CK=$1 BT=$2 t=$3 GO WB WV B=/tmp/tn_blk_$2_$3.txt
  set -- ${V[$t]}
  env SE_GUARD=1 SELL_ORDER=held "$@" DAY=llall NW=126 $COMMON TAGX=_${BT}_$t $PY live/top_bench.py ckpt/$CK.pt > logs/tn_ll_${BT}_$t.log 2>&1
  env SE_GUARD=1 SELL_ORDER=held "$@" DAY=fresh NW=135 $COMMON TAGX=_${BT}_$t $PY live/top_bench.py ckpt/$CK.pt > logs/tn_top_${BT}_$t.log 2>&1
  env SE_GUARD=1 SELL_ORDER=held "$@" DAY=fresh2 NW=120 $COMMON TAGX=_${BT}_$t $PY live/top_bench.py ckpt/$CK.pt > logs/tn_top2_${BT}_$t.log 2>&1
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
    env SE_GUARD=1 SELL_ORDER=held "$@" NW=78 LOCKSTEP=6 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_${BT}_$t \
      $PY live/bench_multi.py ckpt/$CK.pt:10:0.5 > logs/tn_464_${BT}_$t.log 2>&1
    { echo "== $BT + $t 464"; pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } > $B; cat $B >> $L
  fi
}
chain() { local CK=$1 BT=$2 t; shift 2; for t in "$@"; do run_var $CK $BT $t; done; }
# priority order: sheep (W1) and SE by crop shops (W2) first, then sell order / feed guard, ...
chain ilsp_v13_p3 soh sy7 can sysy cow combo2 tom &
chain ilsp_v13_p3 soh se2 fg str tomD m10 combo &
wait; say "SOH BASE DONE"
chain ilsp_v13_newplan nph sy7 can combo2 &
chain ilsp_v13_newplan nph se2 cow str &
wait; say "NPH (reduced) DONE"
# 3. reactive head-to-heads of v13_p3, held vs canonical: vs the public V94 (scripted, 2 at once) and the live
#    NEURAL_20MW8_SE package (neural opponent ~1.6 GB per worker: one at a time); 40 worlds x 2 seats each
for so in held canonical; do
  NW=80 SE_GUARD=1 SELL_ORDER=$so $PY live/h2h_rules.py ckpt/ilsp_v13_p3.pt $K/agents/ref/V94_2945_main.py 40 tn_ilsp_v13_p3_${so}_vsV94 > logs/tn_h2h_${so}_vsV94.log 2>&1 &
done; wait
for so in held canonical; do
  NW=80 SE_GUARD=1 SELL_ORDER=$so $PY live/h2h_rules.py ckpt/ilsp_v13_p3.pt $K/package/NEURAL_20MW8_SE/main.py 40 tn_ilsp_v13_p3_${so}_vs20mw8 > logs/tn_h2h_${so}_vs20mw8.log 2>&1
done
for ot in vsV94 vs20mw8; do
  for so in held canonical; do grep -v -i warn logs/tn_h2h_${so}_$ot.log | tail -1 | sed "s/^/  $so $ot: /" >> $L; done
  $PY live/pair_cmp.py logs/h2h_tn_ilsp_v13_p3_canonical_$ot.json logs/h2h_tn_ilsp_v13_p3_held_$ot.json 2>/dev/null | sed "s/^/  v13_p3 held vs canonical ($ot) /" | cut -c1-200 >> $L
done
say "FAST batch done"
