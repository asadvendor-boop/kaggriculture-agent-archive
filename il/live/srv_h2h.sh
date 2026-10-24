#!/bin/bash
# Sep 28 ~03:25 PKT: R1 (SELL_ORDER=base) against REACTIVE opponents (il/live/h2h_rules.py): our v13_p3 + SE with the
# canonical vs the premium-first sell order, same 40 fresh worlds x both seats, against (1) the live NEURAL_20MW8_SE
# package (exact uploaded main.py, canonical order baked in) and (2) the scripted race bot M7. Paired on seeds + seats.
# Output: /root/a100_bench.log ("== H2H ...").
exec 5>/tmp/srv_h2h${LOCKX}.lock; flock -n 5 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log; NWH=${NWH:-10}; NS=${NS:-40}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P=$K/package/NEURAL_20MW8_SE/main.py
for opp in ${OPPS:-"$P:vs20mw8" "M7:vsM7"}; do
  o=${opp%%:*}; t=${opp##*:}
  for so in ${SOS:-base canonical}; do
    NW=$NWH SE_GUARD=${SEG:-1} SELL_ORDER=$so $PY live/h2h_rules.py ckpt/${CK:-ilsp_v13_p3}.pt $o $NS ${CKT}${so}_$t 2>&1 | grep -v -i warn | tail -2 > logs/h2h_${CKT}${so}_$t.log
  done
  { echo "== H2H ${CK:-ilsp_v13_p3} SE_GUARD=${SEG:-1} SELL_ORDER ${SOS:-base} against $t (reactive; $NS worlds x 2 seats), paired vs v13_p3+SE canonical${CKT:+ and v13_p3 held}"
    for so in ${SOS:-base}; do grep -h "^h2h" logs/h2h_${CKT}${so}_$t.log | sed "s/^/  /"; done
    for so in ${SOS:-base}; do
      [ "$so$CKT" != canonical ] && $PY live/pair_cmp.py logs/h2h_canonical_$t.json logs/h2h_${CKT}${so}_$t.json 2>/dev/null | sed "s/^/  paired vs v13_p3 canonical /" | cut -c1-200
      [ -f logs/h2h_held_$t.json ] && [ "$so$CKT" != held ] && $PY live/pair_cmp.py logs/h2h_held_$t.json logs/h2h_${CKT}${so}_$t.json 2>/dev/null | sed "s/^/  paired vs v13_p3 held /" | cut -c1-200
    done; } >> $L
done
echo "[$(date -u +%H:%M) UTC] H2H done" >> $L
