#!/bin/bash
# starts RL4 when RL3b's PPO process has exited (GPU memory and CPU workers free again)
cd /Users/asad.ali/kaggriculture/il
while pgrep -f "[p]po_live.py .*rl3b.pt" >/dev/null; do sleep 60; done
sleep 30
NW_RL=16 setsid nohup bash live/srv_rl4.sh > logs/rl4_chain.log 2>&1 < /dev/null &
echo "[$(date -u +%H:%M) UTC] RL4 chain started"
