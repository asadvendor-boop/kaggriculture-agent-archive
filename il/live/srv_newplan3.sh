#!/bin/bash
# Sep 28 ~04:50 PKT: ilsp_v13_newplan (v13_p3 + 1 epoch on DSM + M&M&P&Q seats of Sep 26-27) beat v13_p3 by +39 leader-test
# wins and +39 live-loss wins (our bank +$3.4-13k/game; no test episode in its data). Refresh it with the two teams' newest
# games: (1) their known + newer submissions (/root/top10_new.json, then any seen since Sep 27 in the rows files) ->
# data/tapes_top12 (public endpoints, 12 threads; only what is missing); (2) seats + v6 features for the new games;
# (3) il/data/newplan3 = every DSM / M&M&P&Q seat of Sep 26-28 (benchmark worlds excluded) + the same 2,000 anchors;
# (4) v13_p3 + 1 epoch at 1e-4 -> ckpt/ilsp_v13_newplan3.pt (the eval is queued by hand). Log /root/newplan.log
exec 8>/tmp/srv_newplan3.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; LOG=/root/newplan.log; NW=${NW:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
cp -n data/tapes_top12/index.json /root/top12_index_before_np3.json
$PY - <<'PY'
import json
new = json.load(open("/root/top10_new.json")); known = json.load(open("/root/top10_known.json"))
subs = {t: sorted(set(known.get(t, []) + new.get(t, []))) for t in ("DSM", "M & M & P & Q")}
json.dump(subs, open("/root/np3_known.json", "w")); print(subs)
PY
KNOWN_JSON=/root/np3_known.json SINCE=2026-09-27 $PY il/live/srv_discover_subs.py /root/np3_newer.json >> $LOG 2>&1
$PY - <<'PY'
import json
a = json.load(open("/root/np3_known.json")); b = json.load(open("/root/np3_newer.json"))
for t, ids in b.items():
    a.setdefault(t, []); a[t] = sorted(set(a[t]) | {int(i) for i in ids})
json.dump(a, open("/root/np3_subs.json", "w")); print(a)
PY
say "newplan3: fetching DSM + M&M&P&Q games of $(cat /root/np3_subs.json)"
SUBS_ONLY=/root/np3_subs.json THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_np3.log 2>&1
$PY - >> $LOG 2>&1 <<'PY'
import json, os
R = "/Users/asad.ali/kaggriculture"
old = json.load(open("/root/top12_index_before_np3.json")); new = json.load(open(f"{R}/data/tapes_top12/index.json"))
add = {e: v for e, v in new.items() if e not in old and os.path.exists(f"{R}/data/tapes_top12/{e}.json.gz")}
d = f"{R}/data/tapes_np3"; os.makedirs(d, exist_ok=True)
for e in add:
    if not os.path.lexists(f"{d}/{e}.json.gz"):
        os.symlink(f"{R}/data/tapes_top12/{e}.json.gz", f"{d}/{e}.json.gz")
json.dump(add, open(f"{d}/index.json", "w"), ensure_ascii=False)
days = sorted({v.get("day") for v in add.values() if v.get("day")})
print(f"newplan3: {len(add)} new games of the two teams (days {days})", flush=True)
PY
mkdir -p il/data/np3_all
for w in $(seq 0 $((NW - 1))); do nice -n 5 $PY il/build_dataset3.py il/data/np3_all $w $NW data/tapes_np3 > il/logs/build_np3_w$w.log 2>&1 & done; wait
for w in $(seq 0 $((NW - 1))); do TAPES=data/tapes_np3 SEATS=il/data/np3_all nice -n 5 $PY il/live/srv_build_x6.py $w $NW > il/logs/x6_np3_w$w.log 2>&1 & done; wait
$PY - >> $LOG 2>&1 <<'PY'
import os, glob, json
import numpy as np
K = "/Users/asad.ali/kaggriculture"; R = f"{K}/il/data"; out = f"{R}/newplan3"; os.makedirs(out, exist_ok=True)
voc = json.load(open(f"{R}/team_vocab.json"))
held = set(json.load(open(f"{K}/il/live/fresh_heldout_eps.json")))
held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/ladder_worlds.json"))}
for f in ("top_worlds_fresh_300.json", "top_worlds_fresh2_300.json", "top_worlds_livelosses_300.json", "top_worlds_sevloss_300.json"):
    held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/{f}"))}
n = {"old_newplan": 0, "new_plan_seats": 0, "held_out": 0, "other": 0}
for f in glob.glob(f"{R}/newplan/*.npz"):                  # the first newplan set (984 plan seats + 2,000 anchors)
    b = os.path.basename(f)
    if not os.path.lexists(f"{out}/{b}"): os.symlink(os.path.realpath(f), f"{out}/{b}")
    n["old_newplan"] += 1
for f in glob.glob(f"{R}/np3_all/*.npz"):
    b = os.path.basename(f); ep = int(b.split("_")[0])
    if ep in held or ep % 8 == 3: n["held_out"] += 1; continue
    if not os.path.exists(f"{R}/x6/{b[:-4]}.npy"): continue
    ti = int(np.load(f)["meta"][6]); team = voc[ti] if ti < len(voc) else "?"
    if team in ("DSM", "M & M & P & Q"):
        if not os.path.lexists(f"{out}/{b}"): os.symlink(f, f"{out}/{b}")
        n["new_plan_seats"] += 1
    else:
        n["other"] += 1
print(f"newplan3: {n} -> {len(os.listdir(out))} seats", flush=True)
PY
while pgrep -f "python train6.py" > /dev/null; do sleep 60; done
say "fine-tune: v13_p3 + 1 epoch of newplan3 at 1e-4"
cd il
env SEED=0 MAX_LR=1e-4 INIT=ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 \
  $PY train6.py ckpt/ilsp_v13_newplan3.pt 1 320 data/newplan3 > logs/ilsp_v13_newplan3.log 2>&1
say "NEWPLAN3 DONE: $(grep 'best val loss' logs/ilsp_v13_newplan3.log | tail -1)"
