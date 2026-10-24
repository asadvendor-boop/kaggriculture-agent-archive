#!/bin/bash
# Sep 26 (operator: "test both once again, maybe there was a bug in our judgement"): the wash trade re-tested on the NEURAL
# agent against TODAY's field (September's -170/-296 verdict was measured on scripted agents when the ladder was full of
# wheat-BUYING openers; today's top teams are net wheat sellers). WASH=w: on turns we already SELL wheat / fertilizer, sell w
# more from the shed and buy them back at the end of the order list. One-world check: base +5,231 -> w20 +8,184 (the
# opponent's bank -2.8k), every-turn washing -10,534 (subsidises the opponent's buys). RL24, 464 worlds + both leader tests.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M2=ckpt/rl2_it24.pt
echo "[$(date -u +%H:%M) UTC] wash re-test starts"
for w in 20 40 10; do
  WASH=$w TAGX=_r3_tau0.5_wash$w bench_gov ladder $M2:10:0.5
done
$PY live/pair_cmp.py logs/ladder_rl2_it24_t10_r3_tau0.5_test.json logs/ladder_rl2_it24_t10_r3_tau0.5_wash{10,20,40}_test.json
for w in 20 40; do
  WASH=$w TAGX=_wash$w bench_gov top:10 $M2
  WASH=$w TAGX=_wash$w bench_gov top2:10 $M2
done
$PY live/pair_cmp.py logs/top_fresh_rl2_it24_t10.json logs/top_fresh_rl2_it24_t10_wash{20,40}.json
$PY live/pair_cmp.py logs/top_fresh2_rl2_it24_t10.json logs/top_fresh2_rl2_it24_t10_wash{20,40}.json
echo "[$(date -u +%H:%M) UTC] WASH DONE"
