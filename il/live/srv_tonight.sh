#!/bin/bash
# Sep 28 evening batch (operator: "fix all our weaknesses in tonight's build"; memory live-weaknesses W1-W7), revised after
# NPS's live losses (memory eval-traps-sep28): the SE-OFF choice, not the sell order, lost NPS's games; both SE-ON packages
# would have won 2 of its 3 losses. Rule-layer fixes are stacked on the two SE-on candidates:
#   A = NEURAL_20MW8_SE_SOH  = v13_p3 (live 20MW8 network) + SELL_ORDER=held + SE_GUARD=1        (tag soh)
#   B = NEURAL_20MW8NP_SE_SOH = ilsp_v13_newplan (seed 0)  + SELL_ORDER=held + SE_GUARD=1        (tag nph)
# variants: sy7 = SHEEP_YARN=7 (W1 wool), tom = TOMATO_SHOPS=2 (W6), cow = COW_MILK=3 (W7), fg = FEED_GUARD=1 (W3),
#           can = SELL_ORDER=canonical (control: does held-first cost close games live? 464 2400+ said -5),
#           str = STRAW_UP=4 (TOPVSTOP_A), tomD = DSM-calibrated tomatoes (days 15-18, 4 per shop; TOPVSTOP_B),
#           se2 = SE_SHOPS=2 (W2; base B only), combo = all but se2 / tomD.
# Rulers: ALL our live losses so far (il/live/liveloss_all.py -> DAY=llall, incl. NPS's) + both leader tests, then the 464
# for every variant not worse there (the 464 is the least biased ruler - weigh it most). Smoke tests first (are the extra
# sheep / tomatoes placed?). Sized for the 256-thread box (RT_CPU / RT_CAP / NWL). Output /root/tonight.log.
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
source live/srv_cpu.sh; export CPU_BUDGET=${RT_CPU:-240} BENCH_CAP=${RT_CAP:-200}; NWL=${NWL:-150}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $L; }
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
say "tonight batch starts: $($PY live/liveloss_all.py)"
# 0. baselines on all live losses: the live scripts (20MW8_SE, NPS) and the two bases
SE_GUARD=1 DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_se FRESH=1 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/tn_ll_v13.log 2>&1
SELL_ORDER=held DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_soup3nose FRESH=1 $PY live/top_bench.py ckpt/ilsp_v13_newplan_soup3.pt > logs/tn_ll_nps.log 2>&1
SE_GUARD=1 SELL_ORDER=held DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_soh FRESH=1 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/tn_ll_soh.log 2>&1
SE_GUARD=1 SELL_ORDER=held DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_nph FRESH=1 $PY live/top_bench.py ckpt/ilsp_v13_newplan.pt > logs/tn_ll_nph.log 2>&1
{ echo "== all live losses so far, paired vs the live NEURAL_20MW8_SE (v13_p3 + SE)"
  for f in top_llall_ilsp_v13_newplan_soup3_t10_soup3nose top_llall_ilsp_v13_p3_t10_soh top_llall_ilsp_v13_newplan_t10_nph; do pair "llall" logs/top_llall_ilsp_v13_p3_t10_se.json logs/$f.json; done; } >> $L
# 1. smoke tests: sheep on the 2+ yarn-store live losses, tomatoes / cows on the Planned Economy loss (both bases)
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
say "2+ yarn-store live losses for the sheep smoke test: $EPS"
for b in "ilsp_v13_p3:A" "ilsp_v13_newplan:B"; do
  ck=${b%%:*}; tg=${b##*:}
  EPS=$EPS RULES_ENV="SE_GUARD=1 SELL_ORDER=held SHEEP_YARN=7" $PY live/rule_smoke.py ckpt/$ck.pt 6 llall > logs/tn_smoke_sheep_$tg.log 2>&1 &
  EPS=$EPS RULES_ENV="SE_GUARD=1 SELL_ORDER=held" $PY live/rule_smoke.py ckpt/$ck.pt 6 llall > logs/tn_smoke_nosheep_$tg.log 2>&1 &
  EPS=114547731 RULES_ENV="SE_GUARD=1 SELL_ORDER=held TOMATO_SHOPS=2 COW_MILK=3" $PY live/rule_smoke.py ckpt/$ck.pt 1 llall > logs/tn_smoke_tomato_$tg.log 2>&1 &
  EPS=114547731 RULES_ENV="SE_GUARD=1 SELL_ORDER=held" $PY live/rule_smoke.py ckpt/$ck.pt 1 llall > logs/tn_smoke_notomato_$tg.log 2>&1 &
done; wait
for tg in A B; do
  { echo "== base $tg: sheep smoke test (rule on | off), days 12 / 16 / 20"
    paste -d'|' <(grep -E "== ep|d1[26]:|d20:" logs/tn_smoke_sheep_$tg.log | cut -c1-120) <(grep -E "== ep|d1[26]:|d20:" logs/tn_smoke_nosheep_$tg.log | cut -c1-110)
    echo "== base $tg: tomato/cow smoke test on the Planned Economy loss (rules on | off)"
    paste -d'|' <(grep -E "== ep|d08:|d1[0268]:|d2[24]:" logs/tn_smoke_tomato_$tg.log | cut -c1-130) <(grep -E "== ep|d08:|d1[0268]:|d2[24]:" logs/tn_smoke_notomato_$tg.log | cut -c1-110); } >> $L
done
# 2. the variants on each base (base env: SE_GUARD=1 SELL_ORDER=held)
declare -A V=( [sy7]="SHEEP_YARN=7" [tom]="TOMATO_SHOPS=2" [cow]="COW_MILK=3" [fg]="FEED_GUARD=1" [se2]="SE_SHOPS=2"
               [str]="STRAW_UP=4" [tomD]="TOMATO_SHOPS=2 TOMATO_FROM_DAY=15 TOMATO_PER_SHOP=4" [can]="SELL_ORDER=canonical"
               [combo]="SHEEP_YARN=7 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4" [sysy]="SHEEP_YARN=7 SE_SHOPS=2"
               [combo2]="SHEEP_YARN=7 SE_SHOPS=2 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4" )
declare -A TOT
run_base() {   # run_base <ckpt> <base tag> <variant list>
  local CK=$1 BT=$2 t; shift 2
  for t in "$@"; do
    set -- ${V[$t]}
    env SE_GUARD=1 SELL_ORDER=held "$@" DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_${BT}_$t FRESH=1 \
      $PY live/top_bench.py ckpt/$CK.pt > logs/tn_ll_${BT}_$t.log 2>&1
    ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_${BT}_$t; bench_gov top:10 ckpt/$CK.pt > logs/tn_top_${BT}_$t.log 2>&1 )
    ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_${BT}_$t; bench_gov top2:10 ckpt/$CK.pt > logs/tn_top2_${BT}_$t.log 2>&1 )
    { echo "== $BT + $t (${V[$t]}) paired vs $BT"
      pair "live losses" logs/top_llall_${CK}_t10_$BT.json logs/top_llall_${CK}_t10_${BT}_$t.json
      for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_${CK}_t10_$BT.json logs/top_${d}_${CK}_t10_${BT}_$t.json; done; } >> $L
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
    TOT[${BT}_$t]="$WB->$WV"; echo "  $BT + $t total wins (live losses + both leader tests): $WB -> $WV" >> $L
    if [ "$GO" = 1 ]; then
      ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_r3_tau0.5_${BT}_$t; bench_gov ladder ckpt/$CK.pt:10:0.5 > logs/tn_464_${BT}_$t.log 2>&1 )
      { echo "== $BT + $t 464"; pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } >> $L
    fi
  done
}
# priority first (the live losses cluster on sheep / SE-in-no-crop-towns; `can` settles the sell order), both bases,
# then the rest - if time runs short the batch can be stopped after the first two lines
# 14:20 PKT Sep 28: live after ~44 games SOH (v13_p3) 2,696 vs NPSE (new-plan net) 2,645 -> the whole soh base first
run_base ilsp_v13_p3 soh can sy7 se2 sysy combo2
say "PRIORITY VARIANTS DONE (soh)"
run_base ilsp_v13_p3 soh cow tomD tom str fg combo
say "SOH BASE DONE"
run_base ilsp_v13_newplan nph can sy7 se2 sysy combo2
run_base ilsp_v13_newplan nph cow tomD tom str fg combo
# 3. reactive head-to-heads against several styles (the least biased check after the 464): both networks x both sell
#    orders vs the public V94 (2,945) agent and the live NEURAL_20MW8_SE package; 40 worlds x 2 seats each
for opp in "$K/agents/ref/V94_2945_main.py:vsV94" "$K/package/NEURAL_20MW8_SE/main.py:vs20mw8"; do
  o=${opp%%:*}; ot=${opp##*:}
  for ck in ilsp_v13_p3 ilsp_v13_newplan; do
    for so in held canonical; do
      NW=${NWH:-80} SE_GUARD=1 SELL_ORDER=$so $PY live/h2h_rules.py ckpt/$ck.pt $o 40 tn_${ck}_${so}_$ot 2>&1 | grep -v -i warn | tail -1 | sed "s/^/  /" >> $L
    done
    $PY live/pair_cmp.py logs/h2h_tn_${ck}_canonical_$ot.json logs/h2h_tn_${ck}_held_$ot.json 2>/dev/null | sed "s/^/  $ck held vs canonical ($ot) /" | cut -c1-200 >> $L
  done
done
say "tonight batch done: $(for k in "${!TOT[@]}"; do printf '%s %s  ' $k ${TOT[$k]}; done)"
