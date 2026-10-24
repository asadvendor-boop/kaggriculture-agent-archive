#!/bin/bash
# Visible IL-sprint monitor (Mac), Sep 27 evening: every 2 minutes prints the new lines of the sprint log (models done,
# queue empty), of the bench log (leader tests paired vs NEURAL_20MW5_SE, 464 results) and a GPU summary.
# Imitation training is data-loader bound (70-85% utilization is normal), so the GPU line flags only an average below
# 60% or a GPU at 0% in all of its recent samples (= idle GPU).
# Usage: watch_il.sh <port> <host>
P=$1; H=$2; K=/Users/asad.ali/kaggriculture
D=$K/il/logs/il_sprint; mkdir -p $D/.n
while true; do
  for a in 1 2 3; do rsync -az --timeout=60 -e "ssh -o ConnectTimeout=15 -p $P" \
      root@$H:/root/il_sprint.log root@$H:/root/il_bench.log root@$H:/root/gpu_mon.log $D/ 2>/dev/null && break; sleep 5; done
  for f in il_sprint.log il_bench.log; do
    o=$(cat $D/.n/$f 2>/dev/null || echo 0); n=$(wc -l < $D/$f 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -gt "$o" ] && tail -n +$((o + 1)) $D/$f | grep -v "^  *$" | sed "s/^/[${f%.log}] /" | cut -c1-260
    echo $n > $D/.n/$f
  done
  awk '$2 ~ /^gpu/ {u=$4; sub(/%/,"",u); n++; s+=u; if (u+0<mn || n==1) mn=u+0; c[$2]++; if (u+0>0) busy[$2]=1}
       END {idle=0; for (g in c) if (!(g in busy)) idle++
            if (n) printf "[gpu] last 2 min: util avg %.0f%% (min %d%%)%s%s\n", s/n, mn, (s/n<60 ? "  <-- LOW UTILIZATION" : ""), (idle ? "  <-- IDLE GPUs: " idle : "")}' \
      <(grep -v FLAG $D/gpu_mon.log 2>/dev/null | tail -48)
  sleep 120
done
