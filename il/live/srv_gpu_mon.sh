#!/bin/bash
# GPU monitor on the sprint machine: every 20 s one line per GPU (util %, memory used/total); FLAG lines when a GPU in
# use is < 80% busy for 3 samples in a row, or memory > 90% (OOM risk).  Log: /root/gpu_mon.log
declare -A low
while true; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=', ' read i u m t; do
    echo "$(date -u +%H:%M:%S) gpu$i util $u% mem $m/$t MiB"
    [ "$m" -gt 1000 ] && [ "$u" -lt 80 ] && echo "$(date -u +%H:%M:%S) FLAG gpu$i low utilization $u%"
    [ $((m * 100 / t)) -gt 90 ] && echo "$(date -u +%H:%M:%S) FLAG gpu$i memory $((m * 100 / t))% (OOM risk)"
  done
  sleep 20
done
