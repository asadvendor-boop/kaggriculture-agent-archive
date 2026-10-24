#!/bin/bash
# Visible sprint monitor (Mac): every 2 minutes prints the new lines of the two JAX runs' logs (iteration timing, KL,
# evaluations), the checkpoint tester's results and the GPU monitor's flags from the sprint machine.
# Usage: watch_sprint.sh <port> <host> [runA_name] [runB_name]
P=$1; H=$2; A=${3:-sprint_A}; B=${4:-sprint_B}; K=/Users/asad.ali/kaggriculture
D=$K/il/logs/sprint; mkdir -p $D/.n
while true; do
  for a in 1 2 3; do rsync -az --timeout=60 -e "ssh -o ConnectTimeout=15 -p $P" \
      root@$H:$K/kagjax/runs/$A/log.txt root@$H:$K/kagjax/runs/$B/log.txt root@$H:$K/il/logs/ckpt_eval.log root@$H:/root/gpu_mon.log $D/ 2>/dev/null && break; sleep 5; done
  for f in $D/log.txt $D/ckpt_eval.log $D/gpu_mon.log; do [ -f $f ] || continue; done
  # the two run logs have the same file name -> fetch separately
  for r in $A $B; do
    rsync -az --timeout=60 -e "ssh -o ConnectTimeout=15 -p $P" root@$H:$K/kagjax/runs/$r/log.txt $D/$r.log 2>/dev/null
    o=$(cat $D/.n/$r 2>/dev/null || echo 0); n=$(wc -l < $D/$r.log 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -gt "$o" ] && tail -n +$((o + 1)) $D/$r.log | grep -E "^it |^\[eval|exported|Traceback|Error|NaN|nan" | sed "s/^/[$r] /" | cut -c1-260
    echo $n > $D/.n/$r
  done
  # GPU summary over the last 2 minutes (6 samples x 8 GPUs): brief dips between training rounds are normal
  awk 'NR > 0 && $2 ~ /^gpu/ {u=$4; sub(/%/,"",u); split($6,m,"/"); n++; s+=u; if (u+0<mn || n==1) mn=u+0; if (m[1]+0>mx) mx=m[1]+0; t=m[2]+0} END {if (n) printf "[gpu] last 2 min: util avg %.0f%% (min %d%%), memory peak %.1f of %.1f GB%s\n", s/n, mn, mx/1024, t/1024, (s/n<80 ? "  <-- LOW UTILIZATION" : "") (mx>0.9*t ? "  <-- MEMORY NEAR LIMIT" : "")}' <(tail -48 $D/gpu_mon.log 2>/dev/null | grep -v FLAG)
  for f in ckpt_eval.log; do
    o=$(cat $D/.n/$f 2>/dev/null || echo 0); n=$(wc -l < $D/$f 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -gt "$o" ] && tail -n +$((o + 1)) $D/$f | grep -v "^  *$" | grep -E "testing|done|==|vs |FLAG|n 464|n 138|n 122" | sed "s/^/[$f] /" | cut -c1-260
    echo $n > $D/.n/$f
  done
  sleep 120
done
