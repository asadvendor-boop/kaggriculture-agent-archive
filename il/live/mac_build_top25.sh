#!/bin/bash
# Sep 30: one consolidated winner set of the top-25 teams since Sep 26 (each team its own style slot via VOCAB), excluding
# games already in fresh2/dsm_ft2 and held-out test worlds -> il/data/win_top25; DSM-only subset -> il/data/win_dsm30
K=/Users/asad.ali/kaggriculture; W=$K/il/ilwin; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd $K
F="--min-opp-rating 2400 --min-margin 500 --fail-mask 1 --strip-churn 1"; out=$K/il/data/win_top25; NW=3
T="data/tapes_top25:L data/tapes_top25b:L data/tapes_top30:L data/tapes_top30b:L data/tapes_zip27:L data/tapes_zip28:L data/tapes_top10d:L data/tapes_lead29:L data/tapes_today:L data/tapes_rep29:L data/tapes_top12_recent:L data/tapes_spec_more_recent:L"
for i in $(seq 0 $((NW - 1))); do
  VOCAB=$K/il/data/team_vocab_sep30.json TORCH_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $PY $W/build_winner_ds.py --out $out --tapes $T $F \
    --winner-names $K/il/live/top25_names.json --exclude-manifest $K/il/data/exclude_fresh2_dsm.json --worker $i --workers $NW > $K/il/logs/win_top25_w$i.log 2>&1 &
done
wait
grep -h "selected\|per dir" $K/il/logs/win_top25_w0.log | head -2 | cut -c1-600; grep -h "done:" $K/il/logs/win_top25_w*.log
VOCAB=$K/il/data/team_vocab_sep30.json $PY $W/build_winner_ds.py --out $out --merge > $K/il/logs/win_top25_merge.log 2>&1
$PY $W/check_ds.py $out 2>&1 | tail -3
$PY - <<'PY'
import json, os, collections
K = "/Users/asad.ali/kaggriculture/il/data"; src = f"{K}/win_top25"; dst = f"{K}/win_dsm30"
G = json.load(open(f"{src}/manifest.json"))["games"]; G = G if isinstance(G, list) else list(G.values())
c = collections.Counter(g["names"][g["winner"]] for g in G); print("winning seats per team:", dict(c.most_common()))
os.makedirs(f"{dst}/seats", exist_ok=True); os.makedirs(f"{dst}/x6", exist_ok=True); keep = []
for g in G:
    if g["true_team"] != 10: continue
    n = f"{g['ep']}_{g['winner']}"
    for sub, ext in (("seats", ".npz"), ("x6", ".npy")):
        s, t = f"{src}/{sub}/{n}{ext}", f"{dst}/{sub}/{n}{ext}"
        if os.path.exists(s) and not os.path.exists(t): os.symlink(s, t)
    keep.append(g)
json.dump({"summary": {"from": "win_top25", "true_team": 10, "games": len(keep)}, "games": keep}, open(f"{dst}/manifest.json", "w"))
print("win_dsm30 (DSM winning seats, slot 10):", len(keep))
PY
