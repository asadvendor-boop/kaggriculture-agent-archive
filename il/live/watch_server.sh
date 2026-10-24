#!/bin/bash
# LOCAL watcher of the A100 server (operator, Sep 25): every 10 minutes copies the server's chain logs into il/logs/server/
# and appends every NEW line (training milestones, benchmark results, failures) to il/logs/server_watch.log, plus a
# heartbeat (GPU use, running trainers, hours left before the 24-hour stop at 11:27 UTC Sep 26 = 16:27 PKT).
cd /Users/asad.ali/kaggriculture; mkdir -p il/logs/server
S="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 13692 root@<SERVER_IP>"
while true; do
  for L in step3.log chain2.log fresh_train.log fresh_data.log dsm_open.log gapfix.log rl.log more_seeds.log more_seeds_t.log ppo_rl1.log rl2.log ppo_rl2.log extra_t.log; do
    $S "cat /Users/asad.ali/kaggriculture/il/logs/$L" 2>/dev/null | grep -v -E "Welcome to vast|Have fun|AI agents|authentication fails" > il/logs/server/$L.new
    if [ -s il/logs/server/$L.new ]; then
      touch il/logs/server/$L
      diff il/logs/server/$L il/logs/server/$L.new | grep '^>' | sed "s/^> /[server $L] /"
      mv il/logs/server/$L.new il/logs/server/$L
    fi
  done
  hb=$($S 'echo "GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader), trainers $(ps -eo args | grep -c "^/venv/main/bin/python train6[.]py")"' 2>/dev/null | grep -v -E "Welcome to vast|Have fun|AI agents|authentication fails")
  left=$(( (1790422020 - $(date +%s)) / 60 ))
  echo "[$(date +%H:%M) PKT] server heartbeat: ${hb:-UNREACHABLE}; $((left / 60))h$((left % 60))m left before the 24-h stop"
  sleep 600
done >> il/logs/server_watch.log 2>&1
