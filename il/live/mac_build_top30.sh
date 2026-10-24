#!/bin/bash
# Sep 30: build today's top-10 leader games (data/tapes_top30, since Sep 29 11:00 UTC) with the win29t recipe -> il/data/win30t
K=/Users/asad.ali/kaggriculture; W=$K/il/ilwin; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd $K
F="--min-opp-rating 2400 --min-margin 500 --fail-mask 1 --strip-churn 1"; out=$K/il/data/win30t
for i in 0 1; do
  TORCH_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $PY $W/build_winner_ds.py --out $out --tapes data/tapes_top30:L $F --winner-names $K/il/live/top30_names.json \
    --exclude-manifest $K/il/data/win29/manifest.json $K/il/data/win29t/manifest.json --worker $i --workers 2 > $K/il/logs/win30t_w$i.log 2>&1 &
done
wait
grep -h "selected\|per dir" $K/il/logs/win30t_w0.log | head -2; grep -h "done:" $K/il/logs/win30t_w*.log
$PY $W/build_winner_ds.py --out $out --merge > $K/il/logs/win30t_merge.log 2>&1
$PY $W/check_ds.py $out 2>&1 | tail -5
