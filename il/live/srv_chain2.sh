#!/bin/bash
# Replaces chain_server.sh from its step-3 benchmarks on (operator, Sep 25): the 10M model is now tested on the FRESH data
# (srv_fresh_train.sh) instead of a step 3 on the old union, so the GPU goes to the fresh runs.  Waits for the 10M step-2
# benchmark that chain_server.sh started (it keeps running and logging to chain_server.log), then benchmarks step 3
# (Vadim fine-tunes, team 5) through the CPU governor (srv_cpu.sh: >= 2 cores stay free, one benchmark at a time).
cd /Users/asad.ali/kaggriculture/il
source live/srv_cpu.sh
while pgrep -f "live/bench_multi.py ckpt/s2_big10" > /dev/null; do sleep 30; done
until grep -q "ALL DONE" logs/step3.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M) UTC] benchmarking step 3 (1.33M / 4M, seeds 0-2)"
bench_gov ladder ckpt/bc_v6c_all_vadim_s0.pt:5:0.5 ckpt/bc_v6c_all_vadim_s1.pt:5:0.5 ckpt/bc_v6c_all_vadim_s2.pt:5:0.5 ckpt/bc_v6g_all_vadim_s0.pt:5:0.5 ckpt/bc_v6g_all_vadim_s1.pt:5:0.5 ckpt/bc_v6g_all_vadim_s2.pt:5:0.5
echo "[$(date -u +%H:%M) UTC] CHAIN2 DONE"
