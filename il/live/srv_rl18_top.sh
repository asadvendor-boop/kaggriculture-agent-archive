#!/bin/bash
# Sep 26: Codex reported RL18 beating RL24 12-4 head-to-head (8 worlds). Head-to-head between two versions is not our
# criterion (operator rule: the 2,400+ games and the leader tests), so RL18 (rl2_it18) gets both leader tests next to RL24.
cd /Users/asad.ali/kaggriculture/il
source live/srv_cpu.sh
until [ "$(stat -c %s ckpt/rl2_it18.pt 2>/dev/null)" = "241869979" ]; do sleep 30; done
bench_gov top:10 ckpt/rl2_it18.pt ckpt/rl2_it24.pt
until [ -f live/top_worlds_fresh2_300.json ]; do sleep 60; done
bench_gov top2:10 ckpt/rl2_it18.pt ckpt/rl2_it24.pt
/venv/main/bin/python live/pair_cmp.py logs/top_fresh_rl2_it24_t10.json logs/top_fresh_rl2_it18_t10.json
/venv/main/bin/python live/pair_cmp.py logs/top_fresh2_rl2_it24_t10.json logs/top_fresh2_rl2_it18_t10.json
echo "[$(date -u +%H:%M) UTC] RL18 LEADER TESTS DONE"
