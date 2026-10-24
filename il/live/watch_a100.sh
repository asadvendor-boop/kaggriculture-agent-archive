#!/bin/bash
# Overnight monitor (Mac, Sep 27-28) for the Biohub A100 box: every 3 minutes prints the NEW lines of the box's progress
# logs - the top-10 data build (/root/fresh3.log), the imitation runs (/root/a100_il.log), the tests (/root/a100_bench.log)
# the anti-specialist RL run (iterations + evaluations; stopped Sep 27 21:54 UTC), the new-plan fine-tune (/root/newplan.log), the live poller of both active submissions (/root/live_poll.log:
# every new result, specialist-bank additions) and the specialist-games fetch (/root/spec_more.log) - plus a GPU line when
# the GPU sits idle.
# Usage: watch_a100.sh      (ssh -p 42201 root@<SERVER_IP>)
D=/Users/asad.ali/kaggriculture/il/logs/a100; mkdir -p $D/.n
S="ssh -o ConnectTimeout=20 -p 42201 root@<SERVER_IP>"
while true; do
  for a in 1 2 3; do rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -p 42201" root@<SERVER_IP>:/root/fresh3.log \
      root@<SERVER_IP>:/root/a100_il.log root@<SERVER_IP>:/root/a100_bench.log \
      root@<SERVER_IP>:/root/live_poll.log root@<SERVER_IP>:/root/spec_more.log \
      root@<SERVER_IP>:/root/specft.log root@<SERVER_IP>:/root/rl_refresh.log root@<SERVER_IP>:/root/newplan.log \
      $D/ 2>/dev/null && break; sleep 5; done
  RUNX=$($S 'ls -td /Users/asad.ali/kaggriculture/kagjax/runs/antispec*/ 2>/dev/null | head -1' 2>/dev/null | grep antispec | xargs basename 2>/dev/null)
  [ -n "$RUNX" ] && rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -p 42201" root@<SERVER_IP>:/Users/asad.ali/kaggriculture/kagjax/runs/$RUNX/log.txt $D/rl_$RUNX.log 2>/dev/null
  for f in fresh3.log a100_il.log a100_bench.log live_poll.log spec_more.log specft.log rl_refresh.log newplan.log rl_$RUNX.log; do
    o=$(cat $D/.n/$f 2>/dev/null || echo 0); n=$(wc -l < $D/$f 2>/dev/null | tr -d ' '); n=${n:-0}
    if [ "$n" -gt "$o" ]; then
      if [ "${f#rl_}" != "$f" ]; then tail -n +$((o + 1)) $D/$f | grep -E "^it [0-9]+ |^\[eval|Traceback|Error|STOP|nan|NaN" | sed "s/^/[${f%.log}] /" | cut -c1-240
      else tail -n +$((o + 1)) $D/$f | grep -v "^  *$" | sed "s/^/[${f%.log}] /" | cut -c1-240; fi
    fi
    echo $n > $D/.n/$f
  done
  g=$($S "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" 2>/dev/null | grep -E "^[0-9]")
  [ -n "$g" ] && [ "${g%%,*}" -lt 20 ] && echo "[gpu] utilization ${g%%,*}% (memory ${g##*, } MiB)  <-- GPU IDLE?"
  sleep 180
done
