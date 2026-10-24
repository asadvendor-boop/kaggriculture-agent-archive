"""Sep 28 (light planning, step 1): when does the third field (SE) pay for the new-plan network? For every world of the
four rulers, pair SE-on (nph) vs SE-off (npnoseh) margins of the same network, and read what is visible at the decision
point (day 10 hour 12) from an exact replay of the recorded game: the town's unlocked shops (forced to the recorded
sequence in our runs, so identical), plus the shops unlocked by the end, and the taped rival's lands / herd / cash then.
Writes il/logs/se_world_features.json and prints the split of the SE-off minus SE-on margin by feature.
Engine replays only (no network): ~0.02 s per world.   Usage: python3 il/live/se_world_features.py"""
import sys, json, gzip, collections
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/engine/v2")
from harness import kagsim_trace, PASS
K = "/Users/asad.ali/kaggriculture"; B = f"{K}/backup_a100_sep28/il_logs"
RULERS = [("liveloss", f"{K}/backup_a100_sep28/il_live_json/top_worlds_sevloss_300.json", "top_sevloss_ilsp_v13_newplan_t10_{}.json"),
          ("fresh", f"{K}/il/live/top_worlds_fresh_300.json", "top_fresh_ilsp_v13_newplan_t10_{}.json"),
          ("fresh2", f"{K}/il/live/top_worlds_fresh2_300.json", "top_fresh2_ilsp_v13_newplan_t10_{}.json"),
          ("464", f"{K}/il/live/ladder_worlds.json", "ladder_ilsp_v13_newplan_t10_r3_tau0.5_{}_test.json")]
STEP = 10 * 24 + 12


def world_state(path, rival_seat):
    d = json.load(gzip.open(path, "rt")); A = d["actions"]
    g = kagsim_trace.Game(d["seed"], 720, False)
    out = {}
    for t in range(720):
        if t == STEP:
            o = g.observe(rival_seat); f = o["farms"][rival_seat]
            out["shops10"] = list((o.get("town") or {}).get("unlocked_shops") or [])
            out["rival_lands10"] = len(f.get("unlocked_quadrants") or []) - 1
            herd = collections.Counter(t.get("animal") for row in f["tiles"] for t in row if isinstance(t, dict) and t.get("animal"))
            out["rival_herd10"] = dict(herd); out["rival_cash10"] = float(f["money"])
        g.step(A[0][t] if isinstance(A[0][t], dict) else dict(PASS), A[1][t] if isinstance(A[1][t], dict) else dict(PASS))
        if g.done:
            break
    o = g.observe(rival_seat)
    out["shops_end"] = list((o.get("town") or {}).get("unlocked_shops") or [])
    out["rival_lands_end"] = len(o["farms"][rival_seat].get("unlocked_quadrants") or []) - 1
    return out


rows = []
for name, wf, pat in RULERS:
    W = json.load(open(wf))
    try:
        on = json.load(open(f"{B}/{pat.format('nph')}")); off = json.load(open(f"{B}/{pat.format('npnoseh')}"))
    except FileNotFoundError as e:
        print(name, "missing", e); continue
    key = (lambda r: int(r["ep"])) if name == "464" else (lambda r: (int(r["ep"]), int(r["seat"])))   # 464: one row per ep
    ON = {key(r): r for r in on}; OFF = {key(r): r for r in off}
    for w in W:
        k = key(w)
        if k not in ON or k not in OFF:
            continue
        us = int(w["seat"]) if "seat" in w else None
        rival = 1 - us if us is not None else 1
        try:
            st = world_state(w["path"], rival)
        except Exception as e:
            continue
        rows.append(dict(ruler=name, ep=int(w["ep"]), seat=us, opp=w.get("opp"), rating=float(ON[k].get("opp_rating") or 0),
                         on=ON[k]["margin"], off=OFF[k]["margin"], **st))
json.dump(rows, open(f"{K}/il/logs/se_world_features.json", "w"))
print(f"{len(rows)} paired worlds")
d = np.array([r["off"] - r["on"] for r in rows])
flip_on = np.array([(r["on"] > 0) and (r["off"] <= 0) for r in rows]); flip_off = np.array([(r["off"] > 0) and (r["on"] <= 0) for r in rows])
print(f"all: SE-off minus SE-on mean {d.mean():+,.0f} (se {d.std() / np.sqrt(len(d)):,.0f}); worlds only SE-on wins {flip_on.sum()}, only SE-off wins {flip_off.sum()}")
def split(label, mask):
    m = np.array(mask)
    for v, s in ((True, m), (False, ~m)):
        if s.sum() >= 10:
            print(f"  {label} = {v!s:5s} n {int(s.sum()):3d} | off-on {d[s].mean():>+7,.0f} (se {d[s].std() / np.sqrt(s.sum()):,.0f}) | only-on wins {int(flip_on[s].sum()):2d} only-off wins {int(flip_off[s].sum()):2d}")
shop_types = sorted({s for r in rows for s in r["shops10"]} | {s for r in rows for s in r["shops_end"]})
print("shops seen:", shop_types)
for s in shop_types:
    split(f"{s} open by day 10", [s in r["shops10"] for r in rows])
    split(f"{s} open by the end", [s in r["shops_end"] for r in rows])
n10 = np.array([len(r["shops10"]) for r in rows]); nend = np.array([len(r["shops_end"]) for r in rows])
for c in sorted(set(n10)):
    split(f"shops open by day 10 == {c}", n10 == c)
for c in sorted(set(nend)):
    split(f"shops by the end == {c}", nend == c)
split("rival bought land 3 by the end", [r["rival_lands_end"] >= 3 for r in rows])
split("rival has >= 3 lands at day 10.5", [r["rival_lands10"] >= 3 for r in rows])
split("rival sheep >= 6 at day 10", [r["rival_herd10"].get("SHEEP", 0) >= 6 for r in rows])
split("rival rating >= 2400", [r["rating"] >= 2400 for r in rows])
