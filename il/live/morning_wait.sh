#!/bin/bash
# Waits (every 5 min) until every overnight server job has finished, then exits with a one-line summary - the exit is the
# signal to run the morning handoff (il/live/morning_handoff.sh, then stop the instance) if the operator has not written.
# Done = RL5 chain done, 5-pass weighted 20M chain done, JAX-run evaluation done or skipped, 5-pass 20M + SE done, and no sub-agent process
# left on the server (agent jobs are pinned to cores 56-63 by kagjax/srv_jax.sh, or wait on its GPU lock).  Hard deadline 04:30 UTC (09:30 PKT): then it exits anyway and reports what is unfinished.
P=${1:-55611}; H=${2:-<SERVER_IP>}
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $P root@$H"
while true; do
  s=$($SSH 'cd /Users/asad.ali/kaggriculture/il;
    a=0; grep -q "RL5 CHAIN DONE" logs/rl5_chain.log 2>/dev/null && a=1
    b=0; grep -q "5-PASS CHAIN DONE" logs/g20w5.log 2>/dev/null && b=1
    c=0; grep -q -E "JAXEVAL (DONE|SKIPPED)" logs/jaxeval.log 2>/dev/null && grep -q "G20W5 SE DONE" logs/g20w5se.log 2>/dev/null && c=1
    k1=$(for p in $(ps -eo pid=); do taskset -pc $p 2>/dev/null; done | grep -c "list: 56-63$")
    k2=$(ps -eo args | grep -c "kagjax_gpu[.]lock")
    k=$((k1 + k2))
    echo "$a $b $c $k"' 2>/dev/null | tail -1)
  now=$(date -u +%H%M)
  if [ -n "$s" ]; then
    set -- $s
    [ "$1$2$3" = "111" ] && [ "$4" = 0 ] && { echo "[$(date -u +%H:%M) UTC] ALL SERVER JOBS DONE (rl5 $1, g20w5 $2, jaxeval $3, agent procs $4)"; exit 0; }
  fi
  if [ "$now" -ge 0430 ] && [ "$now" -lt 1200 ]; then
    echo "[$(date -u +%H:%M) UTC] DEADLINE REACHED; status rl5/g20w5/jaxeval/agent-procs = ${s:-server unreachable}"; exit 0
  fi
  sleep 300
done
