#!/bin/bash
# Sep 26, operator-approved action 4 ("target the matchups we lose"): the two runtime rules from the live-loss study
# (il/live/land_sheep_diag.py: 8 of the current models' 14 live losses never bought the 3rd land; big sheep buys on day 10)
#   se    = SE guard: 3rd land from day 10 hour 12 to day 13 when 2 lands are owned and cash >= $4,300
#   sec12 = SE guard + sheep cap 12 (no sheep beyond 12 until the 3rd land is bought)
#   c15   = sheep cap 15 ONLY (Kimi's priority rule "field-4 before sheep-16": never forces a land buy; the saved cash lets
#           the model buy the field itself - a blanket land buy was net-negative in Kimi's scripted test R3X, and in our
#           replay of the Driz Lo loss the forced buy made it worse)       c12 = sheep cap 12 only
# on both live models (fr_t_dsm_s0 = 56559691, rl2_it24 = 56571775), paired against the unchanged models on the SAME
# worlds.  Rulers: (1) the 464 ladder worlds, (2) all our live games replayed (wins too), (3) the held-out fresh
# leader benchmark (300 worlds).  Keep a rule only if it helps and regresses on none.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M1=ckpt/fr_t_dsm_s0.pt; M2=ckpt/rl2_it24.pt
echo "[$(date -u +%H:%M) UTC] rules chain starts"
SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder $M1:10:0.5 $M2:10:0.5
SE_GUARD=1 SHEEP_CAP=12 TAGX=_r3_tau0.5_sec12 bench_gov ladder $M1:10:0.5 $M2:10:0.5
SHEEP_CAP=15 TAGX=_r3_tau0.5_c15 bench_gov ladder $M1:10:0.5 $M2:10:0.5
SHEEP_CAP=12 TAGX=_r3_tau0.5_c12 bench_gov ladder $M1:10:0.5 $M2:10:0.5
for m in fr_t_dsm_s0 rl2_it24; do
  $PY live/pair_cmp.py logs/ladder_${m}_t10_r3_tau0.5_test.json logs/ladder_${m}_t10_r3_tau0.5_{se,sec12,c15,c12}_test.json
done
# (2) live games: each model on the games it played itself (56559691 -> tapes_n10md, 56571775 -> tapes_n10mr)
live_gov() {   # $1 ckpt  $2 tag  $3 tape dir ; env passes through
  exec 9>/tmp/srv_bench.lock; flock 9
  while [ "$(cpu_free)" -lt 6 ]; do sleep 60; done
  local n=$(cpu_free); [ $n -gt 40 ] && n=40
  NW=$n LOCKSTEP=6 MIDNIGHT=1 FINALFIX=1 HOMING=1 $PY live/live_replay.py $1 10 $2 $3 2>&1 | grep --line-buffered -v Warning
  flock -u 9; exec 9>&-
}
T1=/Users/asad.ali/kaggriculture/data/tapes_n10md; T2=/Users/asad.ali/kaggriculture/data/tapes_n10mr
live_gov $M1 n10md_base $T1; live_gov $M2 n10mr_base $T2
SE_GUARD=1 live_gov $M1 n10md_se $T1; SE_GUARD=1 live_gov $M2 n10mr_se $T2
SE_GUARD=1 SHEEP_CAP=12 live_gov $M1 n10md_sec12 $T1; SE_GUARD=1 SHEEP_CAP=12 live_gov $M2 n10mr_sec12 $T2
SHEEP_CAP=15 live_gov $M1 n10md_c15 $T1; SHEEP_CAP=15 live_gov $M2 n10mr_c15 $T2
SHEEP_CAP=12 live_gov $M1 n10md_c12 $T1; SHEEP_CAP=12 live_gov $M2 n10mr_c12 $T2
for m in n10md n10mr; do $PY live/pair_cmp.py logs/live_${m}_base.json logs/live_${m}_{se,sec12,c15,c12}.json; done
# (3) held-out fresh leader benchmark
bench_gov top:10 $M2
SE_GUARD=1 TAGX=_se bench_gov top:10 $M1 $M2
SE_GUARD=1 SHEEP_CAP=12 TAGX=_sec12 bench_gov top:10 $M1 $M2
SHEEP_CAP=15 TAGX=_c15 bench_gov top:10 $M1 $M2
SHEEP_CAP=12 TAGX=_c12 bench_gov top:10 $M1 $M2
for m in fr_t_dsm_s0 rl2_it24; do
  $PY live/pair_cmp.py logs/top_fresh_${m}_t10.json logs/top_fresh_${m}_t10_{se,sec12,c15,c12}.json
done
echo "[$(date -u +%H:%M) UTC] RULES DONE"
