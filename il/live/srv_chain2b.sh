#!/bin/bash
# Step-3 benchmarks as soon as each size is ready (operator, Sep 25; replaces srv_chain2.sh, which waited for all):
# the three 1.33M runs (done) now, the three 4M runs after "ALL DONE". Vadim fine-tunes -> team 5. CPU governor.
cd /Users/asad.ali/kaggriculture/il
source live/srv_cpu.sh
echo "[$(date -u +%H:%M) UTC] benchmarking step 3, 1.33M (seeds 0-2)"
bench_gov ladder ckpt/bc_v6c_all_vadim_s0.pt:5:0.5 ckpt/bc_v6c_all_vadim_s1.pt:5:0.5 ckpt/bc_v6c_all_vadim_s2.pt:5:0.5
until grep -q "ALL DONE" logs/step3.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M) UTC] benchmarking step 3, 4M (seeds 0-2)"
bench_gov ladder ckpt/bc_v6g_all_vadim_s0.pt:5:0.5 ckpt/bc_v6g_all_vadim_s1.pt:5:0.5 ckpt/bc_v6g_all_vadim_s2.pt:5:0.5
echo "[$(date -u +%H:%M) UTC] CHAIN2 DONE"
