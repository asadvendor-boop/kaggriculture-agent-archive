#!/bin/bash
# Waits until RL3's iteration-6 checkpoint (saved right after its it-6 evaluation) is complete (> 1 min old), then stops
# RL3 (CPU rollouts) and starts RL3b (GPU rollouts) from that checkpoint.
cd /Users/asad.ali/kaggriculture/il
until [ -n "$(find ckpt/rl3_it6.pt -mmin +1 2>/dev/null)" ]; do sleep 30; done
P=$(pgrep -f '^bash live/srv_rl3.sh'); S=$(ps -o sid= -p $P 2>/dev/null | tr -d ' ')
echo "[$(date -u +%H:%M) UTC] rl3_it6.pt ready; stopping RL3 session $S ($(ps -eo sid | awk -v s=$S '$1==s' | wc -l) processes)"
[ -n "$S" ] && pkill -s $S; sleep 5
NW_RL=20 setsid nohup bash live/srv_rl3b.sh > logs/rl3b_chain.log 2>&1 < /dev/null &
echo "[$(date -u +%H:%M) UTC] RL3b chain started"
