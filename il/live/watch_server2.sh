#!/bin/bash
# Visible monitor for the new server (Sep 26, <SERVER_IP>:55611): every 5 minutes the chain logs and result files
# (il/logs/*.log|*.json) are copied back to il/logs/server2/ (resumable, retried - this host drops long connections),
# new progress lines are printed, and new checkpoints (rl3_it*, rf_*) are copied to il/ckpt/ so a host failure loses
# little.   Usage: watch_server2.sh [port] [host]
P=${1:-55611}; H=${2:-<SERVER_IP>}; K=/Users/asad.ali/kaggriculture
D=$K/il/logs/server2; mkdir -p $D/.n
RS() { for a in 1 2 3 4; do rsync -az --partial --timeout=90 -e "ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -p $P" "$@" 2>/dev/null && return 0; sleep 5; done; return 1; }
while true; do
  RS --include='*.log' --include='*.json' --exclude='*' root@$H:$K/il/logs/ $D/ || echo "[$(date -u +%H:%M) UTC] (logs not reachable this round)"
  for f in rl3_switch.log rl3b_chain.log ppo_rl3b.log rl3_chain.log ppo_rl3.log rules4.log wash.log g20w.log g20w5.log g20se.log final1.log pkgcheck.log pkgparity_20mw.log rl4_chain.log ppo_rl4.log rl4se.log rl5_chain.log ppo_rl5.log jaxeval.log g20w5se.log; do
    [ -f $D/$f ] || continue
    o=$(cat $D/.n/$f 2>/dev/null || echo 0); n=$(wc -l < $D/$f | tr -d ' ')
    [ "$n" -gt "$o" ] && tail -n +$((o + 1)) $D/$f | grep -v -i "warning" | sed "s/^/[$f] /" | cut -c1-400
    echo $n > $D/.n/$f
  done
  RS --include='rl3_it*.pt' --include='rl3b_it*.pt' --include='rl4_it*.pt' --include='rl5_it*.pt' --include='rf_*.pt' --include='g20_*.pt' --exclude='*' root@$H:$K/il/ckpt/ $K/il/ckpt/ || true
  sleep 300
done
