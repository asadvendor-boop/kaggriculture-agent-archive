#!/bin/bash
# Sep 26 (replaces the rest of srv_rules.sh, stopped during its cap-12 ladder run: cap 12 already lost to the SE guard alone
# and the 13-15-sheep live games were all wins): the live games and the leader test for base / SE guard / cap 15 on both
# live models (SE + cap 15 runs in srv_rules2.sh). Judged on the 2,400+ games and the leader tests.
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
echo "[$(date -u +%H:%M) UTC] rules3 starts"
T1=/Users/asad.ali/kaggriculture/data/tapes_n10md; T2=/Users/asad.ali/kaggriculture/data/tapes_n10mr
live_gov $M1 n10md_base $T1; live_gov $M2 n10mr_base $T2
SE_GUARD=1 live_gov $M1 n10md_se $T1; SE_GUARD=1 live_gov $M2 n10mr_se $T2
SHEEP_CAP=15 live_gov $M1 n10md_c15 $T1; SHEEP_CAP=15 live_gov $M2 n10mr_c15 $T2
for m in n10md n10mr; do $PY live/pair_cmp.py logs/live_${m}_base.json logs/live_${m}_{se,c15}.json; done
bench_gov top:10 $M2
SE_GUARD=1 TAGX=_se bench_gov top:10 $M1 $M2
SHEEP_CAP=15 TAGX=_c15 bench_gov top:10 $M1 $M2
for m in fr_t_dsm_s0 rl2_it24; do $PY live/pair_cmp.py logs/top_fresh_${m}_t10.json logs/top_fresh_${m}_t10_{se,c15}.json; done
echo "[$(date -u +%H:%M) UTC] RULES3 DONE"
