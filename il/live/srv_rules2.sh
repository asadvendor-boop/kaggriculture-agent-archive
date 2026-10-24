#!/bin/bash
# Sep 26: SE guard + sheep cap 15 (Kimi's threshold), the combination missing from srv_rules.sh (on the 464 worlds the SE
# guard alone was the best single rule, cap 12 hurt, cap 15 alone neutral) - 464 worlds, live games, leader test.
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
M1=ckpt/fr_t_dsm_s0.pt; M2=ckpt/rl2_it24.pt
live_gov() {   # $1 ckpt  $2 tag  $3 tape dir ; env passes through
  exec 9>/tmp/srv_bench.lock; flock 9
  while [ "$(cpu_free)" -lt 6 ]; do sleep 60; done
  local n=$(cpu_free); [ $n -gt 40 ] && n=40
  NW=$n LOCKSTEP=6 MIDNIGHT=1 FINALFIX=1 HOMING=1 $PY live/live_replay.py $1 10 $2 $3 2>&1 | grep --line-buffered -v Warning
  flock -u 9; exec 9>&-
}
echo "[$(date -u +%H:%M) UTC] rules2 (SE guard + sheep cap 15) starts"
SE_GUARD=1 SHEEP_CAP=15 TAGX=_r3_tau0.5_sec15 bench_gov ladder $M1:10:0.5 $M2:10:0.5
for m in fr_t_dsm_s0 rl2_it24; do
  $PY live/pair_cmp.py logs/ladder_${m}_t10_r3_tau0.5_test.json logs/ladder_${m}_t10_r3_tau0.5_{se,c15,sec15}_test.json
done
T1=/Users/asad.ali/kaggriculture/data/tapes_n10md; T2=/Users/asad.ali/kaggriculture/data/tapes_n10mr
SE_GUARD=1 SHEEP_CAP=15 live_gov $M1 n10md_sec15 $T1; SE_GUARD=1 SHEEP_CAP=15 live_gov $M2 n10mr_sec15 $T2
for m in n10md n10mr; do [ -f logs/live_${m}_base.json ] && $PY live/pair_cmp.py logs/live_${m}_base.json logs/live_${m}_{se,c15,sec15}.json; done
SE_GUARD=1 SHEEP_CAP=15 TAGX=_sec15 bench_gov top:10 $M1 $M2
for m in fr_t_dsm_s0 rl2_it24; do
  [ -f logs/top_fresh_${m}_t10.json ] && $PY live/pair_cmp.py logs/top_fresh_${m}_t10.json logs/top_fresh_${m}_t10_{se,c15,sec15}.json
done
echo "[$(date -u +%H:%M) UTC] RULES2 DONE"
