#!/bin/bash
# Sep 28: cancel the 464 run of a model whose leader tests already failed (srv_a100_eval.sh runs it by default).
# Usage: srv_cancel464.sh <ckpt name> "<reason>"   Kills by explicit PID: bench_multi for ckpt/<name>.pt + its workers.
m=$1; why=$2; L=/root/a100_bench.log
while :; do
  p=$(ps -eo pid=,args= | awk -v m="ckpt/$m.pt:" '$2 ~ /python$/ && $3 == "live/bench_multi.py" && index($4, m) == 1 {print $1}')
  [ -n "$p" ] && break
  grep -q "a100 eval done: $m\$" $L && exit 0
  sleep 15
done
kids=$(ps -o pid= --ppid $p); kill $p $kids 2>/dev/null
echo "  ($m 464 cancelled at $(date -u +%H:%M) UTC: $why)" >> $L
