#!/bin/bash
# Sep 26: Codex's JointSeed fix as our rule switch SEED_BUDGET (the engine drops EVERY PLANT of a crop when more units ask
# to plant it than seeds held; keep the first `seeds` requesters). Reproduces Codex's package exactly (ep 112386891:
# +5,231 -> -7,148); on 48 worlds its package was neutral (46 -> 46 wins, +$39 se 556; 2400+ -$354). Full test on RL24:
# seed alone and the full candidate SE guard + cap 15 + seed, on the 464 worlds, the old leader test and today's top-10
# test (+ today's top-10 test for se / sec15, not queued elsewhere).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M2=ckpt/rl2_it24.pt
echo "[$(date -u +%H:%M) UTC] rules4 (seed budget) starts"
SEED_BUDGET=1 TAGX=_r3_tau0.5_seed bench_gov ladder $M2:10:0.5
SE_GUARD=1 SHEEP_CAP=15 SEED_BUDGET=1 TAGX=_r3_tau0.5_sec15seed bench_gov ladder $M2:10:0.5
$PY live/pair_cmp.py logs/ladder_rl2_it24_t10_r3_tau0.5_test.json logs/ladder_rl2_it24_t10_r3_tau0.5_{seed,sec15,sec15seed}_test.json
SEED_BUDGET=1 TAGX=_seed bench_gov top:10 $M2
SE_GUARD=1 SHEEP_CAP=15 SEED_BUDGET=1 TAGX=_sec15seed bench_gov top:10 $M2
for v in se sec15 seed sec15seed; do
  case $v in se) E="SE_GUARD=1";; sec15) E="SE_GUARD=1 SHEEP_CAP=15";; seed) E="SEED_BUDGET=1";; sec15seed) E="SE_GUARD=1 SHEEP_CAP=15 SEED_BUDGET=1";; esac
  env $E TAGX=_$v bash -c "source live/srv_cpu.sh; bench_gov top2:10 $M2"
done
$PY live/pair_cmp.py logs/top_fresh_rl2_it24_t10.json logs/top_fresh_rl2_it24_t10_{se,c15,sec15,seed,sec15seed}.json
$PY live/pair_cmp.py logs/top_fresh2_rl2_it24_t10.json logs/top_fresh2_rl2_it24_t10_{se,sec15,seed,sec15seed}.json
echo "[$(date -u +%H:%M) UTC] RULES4 DONE"
