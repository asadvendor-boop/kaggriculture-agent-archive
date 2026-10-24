#!/bin/bash
# Sep 28 ~17:25 PKT: REACTIVE head-to-heads for the lead candidate (the opponents react to our moves, unlike the taped
# rulers): v13_p3 + held + SE (the live SOH) vs the same + STRADDLE=2, each against the public V94 (scripted) and the live
# NEURAL_20MW8_SE package (neural), 40 worlds x 2 seats = 80 games per run, all 4 runs at once on the CPU.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[$(date -u +%H:%M) UTC] reactive h2h (held vs +STRADDLE=2) vs V94 and the 20MW8 package starts" >> $L
for o in "vsV94:$K/agents/ref/V94_2945_main.py:20:4" "vs20mw8:$K/package/NEURAL_20MW8_SE/main.py:40:2"; do
  IFS=: read ot path nw ls <<< "$o"
  for cfg in "held|" "st2|STRADDLE=2"; do
    t=${cfg%%|*}; envs=${cfg#*|}
    env SE_GUARD=1 SELL_ORDER=held $envs NW=$nw LOCKSTEP=$ls $PY live/h2h_rules.py ckpt/ilsp_v13_p3.pt $path 40 st_${t}_$ot > logs/tn_h2hst_${t}_$ot.log 2>&1 &
    sleep 10
  done
done
wait
{ for ot in vsV94 vs20mw8; do
    for t in held st2; do grep -v -i warn logs/tn_h2hst_${t}_$ot.log | tail -1 | sed "s/^/  h2h $t $ot: /" | cut -c1-200; done
    $PY live/pair_cmp.py logs/h2h_st_held_$ot.json logs/h2h_st_st2_$ot.json 2>/dev/null | sed "s/^/  REACTIVE $ot +STRADDLE=2 vs held: /" | cut -c1-230
  done; } > /tmp/tn_h2hst.txt; cat /tmp/tn_h2hst.txt >> $L
echo "[$(date -u +%H:%M) UTC] reactive h2h done" >> $L
