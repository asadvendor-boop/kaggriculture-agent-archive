#!/bin/bash
# Full 464-world ladder benchmark (tau 0.5, rules on) of each menu-8 model as soon as its training has finished,
# imitated leader's team id; then the leader benchmark (300 seats) for the base and the Vadim / DSM / M&M&P&Q clones.
cd /Users/asad.ali/kaggriculture
run() {  # $1 ckpt tag, $2 team
  LOCKSTEP=6 NW=5 TEAM=$2 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5 nice -n 5 .venv/bin/python il/live/ladder.py eval test il/ckpt/$1.pt 2>&1 | grep -v Warning | grep -v recorded
}
until grep -q "all-data" il/logs/menu8_chain.log 2>/dev/null; do sleep 60; done
run bc_v8m 5
for spec in vadim:5 mmpq:1 dsm:10 tfc:7 mtmr:9 kdew:8 cbf:6 majkel:11 snorlax:47 boey:0 fq:45; do
  L=${spec%%:*}; T=${spec##*:}
  until grep -q "^\[..:..\] $L: best val loss" il/logs/menu8_chain.log il/logs/menu8_more.log 2>/dev/null; do sleep 60; done
  run bc_v8m_$L $T
done
for spec in bc_v8m_vadim:5 bc_v8m_dsm:10 bc_v8m_mmpq:1; do
  NW=5 TEAM=${spec##*:} TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3 nice -n 5 .venv/bin/python il/live/top_bench.py il/ckpt/${spec%%:*}.pt 2>&1 | grep -v Warning
done
echo "[$(date +%H:%M)] menu-8 benchmarks done"
