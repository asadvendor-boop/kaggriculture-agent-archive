#!/bin/bash
# Bug-check control for the DSM-opening result (fr_c_dsm_s1: 365 -> 119 with DSMR0 playing steps 0-143): the same
# handover with a different scripted opening (HighWool, steps 0-143). If this also collapses, the handover code is the
# suspect; if not, DSM's replayed opening itself is the cause.  CPU governor.
cd /Users/asad.ali/kaggriculture/il
source live/srv_cpu.sh
PREFIX_BOT=HighWool PREFIX_STEPS=144 TAGX=_hwopen bench_gov ladder ckpt/fr_c_dsm_s1.pt:10:0.5
echo "[$(date -u +%H:%M) UTC] OPENING CONTROL DONE"
