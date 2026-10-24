#!/bin/bash
# Sep 27 night (operator: "download directly on server, latest submission replays of all top 10 current leaders ... and the
# daily top replay dataset for the last day available"; then weighted IL on it), A100 box. Kaggle PUBLIC endpoints only
# (ListEpisodes + replay JSON, 12 threads, 429 Retry-After honoured; the daily dataset zip) - never the Kaggle token here.
#  1. the leaders' known submissions (current top 10 per the operator's leaderboard, Sep 27 ~22:10 PKT) -> their new games
#     + fresh rows files (every opponent's submission id)
#  2. their NEWER submissions (and the 3 teams new in the top 10) seen as opponents since Sep 26 -> all their games
#  3. the Kaggle daily top-games dataset of 2026-09-25 (the last one published; Sep 26 = 404) -> data/tapes_kd/2026-09-25
#  4. training seats + v6 features for the NEW games only (the 2 new teams added to the team vocabulary: 48 -> 50)
#  5. il/data/fresh3 = fresh2 (Sep 26 build) + the new seats, minus every benchmark game; rating weights
# Progress / "FRESH3 READY": /root/fresh3.log
exec 8>/tmp/srv_fresh3.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; LOG=/root/fresh3.log; NW=${NW:-16}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
cp -n data/tapes_top12/index.json /root/top12_index_before.json
say "start: $($PY -c "import json; print(len(json.load(open('/root/top12_index_before.json'))))") leader games known"
# 0. the current top 10 (operator's screenshot): known submissions; [] = a team new in the top 10
cat > /root/top10_known.json <<'JSON'
{"DECEM": [56558220, 56573934], "DSM": [56564674, 56571979], "Boey": [56521745], "M & M & P & Q": [56553720, 56570008],
 "Vadim Vasilenko": [], "Majkel1337": [56538379], "Unknown Mother-Goose": [56567212], "Fourth Quadrant": [56555482],
 "Just A game on your lips": [], "Anton Tikhonov": []}
JSON
$PY - >> $LOG 2>&1 <<'PY'
import json
p = "/Users/asad.ali/kaggriculture/il/data/team_vocab.json"; v = json.load(open(p))
for t in ("Just A game on your lips", "Anton Tikhonov"):
    if t not in v: v.append(t)
json.dump(v, open(p, "w"), ensure_ascii=False); print(f"team vocabulary: {len(v)} teams", flush=True)
PY
# 1. known submissions (their new games + fresh rows files)
SUBS_ONLY=/root/top10_known.json THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top10_r1.log 2>&1
say "round 1 (known submissions): $(tail -1 il/logs/fetch_top10_r1.log | cut -c1-150)"
# 2. newer submissions of the top 10 (incl. the 3 new teams) seen since Sep 26
KNOWN_JSON=/root/top10_known.json SINCE=2026-09-26 $PY il/live/srv_discover_subs.py /root/top10_new.json >> $LOG 2>&1
SUBS_ONLY=/root/top10_new.json THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top10_r2.log 2>&1
say "round 2 (newer submissions): $(tail -1 il/logs/fetch_top10_r2.log | cut -c1-150)"
# 3. the daily dataset (streamed from the zip, never extracted)
mkdir -p data/kaggle_daily data/tapes_kd; D=2026-09-25; Z=data/kaggle_daily/kaggriculture-episodes-$D.zip
for a in 1 2 3 4 5 6; do
  $PY -c "import zipfile, sys; sys.exit(0 if zipfile.is_zipfile('$Z') else 1)" 2>/dev/null && break
  curl -s -L --retry 3 --retry-delay 30 -o $Z https://www.kaggle.com/api/v1/datasets/download/kaggle/kaggriculture-episodes-$D; sleep 15
done
say "daily $D zip: $(du -h $Z 2>/dev/null | cut -f1)"
$PY il/live/srv_import_day.py $Z $D data/tapes_kd/$D 8 >> $LOG 2>&1
# 4. tape dir with only the NEW top-10 games, then seats + v6 features for new top-10 + daily games
$PY - >> $LOG 2>&1 <<'PY'
import json, os
R = "/Users/asad.ali/kaggriculture"
old = json.load(open("/root/top12_index_before.json")); new = json.load(open(f"{R}/data/tapes_top12/index.json"))
add = {e: v for e, v in new.items() if e not in old and os.path.exists(f"{R}/data/tapes_top12/{e}.json.gz")}
d = f"{R}/data/tapes_top12_new"; os.makedirs(d, exist_ok=True)
for e in add:
    if not os.path.lexists(f"{d}/{e}.json.gz"):
        os.symlink(f"{R}/data/tapes_top12/{e}.json.gz", f"{d}/{e}.json.gz")
json.dump(add, open(f"{d}/index.json", "w"), ensure_ascii=False)
days = sorted({v.get("day") for v in add.values() if v.get("day")})
print(f"new top-10 games: {len(add)} (days {days[:1]}..{days[-1:]})", flush=True)
PY
T="data/tapes_top12_new data/tapes_kd/2026-09-25"
mkdir -p il/data/fresh_all
for w in $(seq 0 $((NW - 1))); do nice -n 5 $PY il/build_dataset3.py il/data/fresh_all $w $NW $T > il/logs/build_fresh3_w$w.log 2>&1 & done; wait
say "seats written: $(tail -q -n1 il/logs/build_fresh3_w*.log | awk '{s+=$8} END {print s}'), non-exact/failed: $(tail -q -n1 il/logs/build_fresh3_w*.log | awk '{s+=$10} END {print s}')"
for w in $(seq 0 $((NW - 1))); do TAPES=$(echo $T | tr ' ' ':') SEATS=il/data/fresh_all nice -n 5 $PY il/live/srv_build_x6.py $w $NW > il/logs/x6_fresh3_w$w.log 2>&1 & done; wait
say "v6 features: $(tail -q -n1 il/logs/x6_fresh3_w*.log | awk '{s+=$5} END {print s}') new"
# 5. fresh3 = fresh2 + new seats, minus every benchmark game (464 ladder worlds, both leader tests, earlier held-out games,
#    and ep % 8 == 3 of the new games, as the leader benchmarks were drawn)
$PY - >> $LOG 2>&1 <<'PY'
import os, glob, json
K = "/Users/asad.ali/kaggriculture"; R = f"{K}/il/data"; out = f"{R}/fresh3"; os.makedirs(out, exist_ok=True)
held = set(json.load(open(f"{K}/il/live/fresh_heldout_eps.json")))
held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/ladder_worlds.json"))}
for f in ("top_worlds_fresh_300.json", "top_worlds_fresh2_300.json"):
    held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/{f}"))}
n = {"fresh2": 0, "new": 0, "held_out": 0, "no_x6": 0}
for f in glob.glob(f"{R}/fresh2/*.npz"):
    b = os.path.basename(f)
    if not os.path.lexists(f"{out}/{b}"): os.symlink(f, f"{out}/{b}")
    n["fresh2"] += 1
for f in glob.glob(f"{R}/fresh_all/*.npz"):
    b = os.path.basename(f); ep = int(b.split("_")[0])
    if ep in held or ep % 8 == 3: n["held_out"] += 1; continue
    if not os.path.exists(f"{R}/x6/{b[:-4]}.npy"): n["no_x6"] += 1; continue
    if not os.path.lexists(f"{out}/{b}"): os.symlink(f, f"{out}/{b}")
    n["new"] += 1
print(f"fresh3: {n} -> {len(os.listdir(out))} seats", flush=True)
PY
SUBS_JSON=/root/top10_new.json $PY il/live/srv_rating_w.py il/data/fresh3 il/data/rating_w_fresh3.json data/tapes_top12 data/tapes_kd/2026-09-25 >> $LOG 2>&1
$PY - >> $LOG 2>&1 <<'PY'
import json          # the fresh2 seats keep the exact weights v13_p3 was trained with (their daily-dataset tapes are not on this box)
R = "/Users/asad.ali/kaggriculture/il/data"; w3 = json.load(open(f"{R}/rating_w_fresh3.json")); w2 = json.load(open(f"{R}/rating_w_fresh2.json"))
n = sum(1 for k in w2 if k in w3); w3.update(w2); json.dump(w3, open(f"{R}/rating_w_fresh3.json", "w"))
print(f"rating weights: {len(w3)} seats ({n} fresh2 seats kept at their Sep 26 weights)", flush=True)
PY
say "FRESH3 READY: $(ls il/data/fresh3 | wc -l) seats, weights il/data/rating_w_fresh3.json"
