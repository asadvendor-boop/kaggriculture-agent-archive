#!/bin/bash
# Sep 29: REACTIVE head-to-heads for the final candidates (the opponents react, unlike the taped rulers): each spec vs the
# public V94 (scripted), the NEURAL_20MW8_SE package and the LIVE ST2 package (NEURAL_20MW8_SOH_ST2) - a candidate must beat
# ST2 itself. 40 worlds x 2 seats per run; our side on the GPU (ROLLOUT_DEVICE=cuda), the package opponents on the CPU
# (~1.6 GB RAM per worker: keep NWH x parallel runs within RAM). Base env SE_GUARD=1 SELL_ORDER=held; put STRADDLE=2 etc. in
# the spec. Usage: srv_h2h_cands.sh <ckpt> "tag|ENV" ["tag2|ENV" ...]   env NWH (16), OPPS (vsV94 vs20mw8 vsST2)
# Results logs/h2h_c_<tag>_<opp>.json; summary + paired lines vs the spec named "st2" (if given) -> /root/tonight.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ROLLOUT_DEVICE=${ROLLOUT_DEVICE:-cuda}
declare -A OP=( [vsV94]=$K/agents/ref/V94_2945_main.py [vs20mw8]=$K/package/NEURAL_20MW8_SE/main.py [vsST2]=$K/package/NEURAL_20MW8_SOH_ST2/main.py )
CK=$1; shift; NW=${NWH:-16}
echo "[$(date -u +%H:%M) UTC] reactive h2h for: $*" >> $L
for spec in "$@"; do
  t=${spec%%|*}; envs=${spec#*|}
  for ot in ${OPPS:-vsV94 vs20mw8 vsST2}; do
    [ -f logs/h2h_c_${t}_$ot.json ] && continue
    env SE_GUARD=1 SELL_ORDER=held $envs NW=$NW LOCKSTEP=$(( (80 + NW - 1) / NW )) $PY live/h2h_rules.py ckpt/$CK.pt ${OP[$ot]} 40 c_${t}_$ot > logs/tn_h2hc_${t}_$ot.log 2>&1
  done
done
{ for spec in "$@"; do t=${spec%%|*}
    for ot in ${OPPS:-vsV94 vs20mw8 vsST2}; do
      grep -v -i warn logs/tn_h2hc_${t}_$ot.log 2>/dev/null | tail -1 | sed "s/^/  h2h $t $ot: /" | cut -c1-200
      [ "$t" != st2 ] && [ -f logs/h2h_c_st2_$ot.json ] && $PY live/pair_cmp.py logs/h2h_c_st2_$ot.json logs/h2h_c_${t}_$ot.json 2>/dev/null | sed "s/^/  REACTIVE $ot $t vs st2: /" | cut -c1-220
    done; done; } > /tmp/tn_h2hc.txt; cat /tmp/tn_h2hc.txt >> $L
echo "[$(date -u +%H:%M) UTC] reactive h2h done" >> $L
