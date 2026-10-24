"""Incremental specialist bank (Sep 27 night, A100 box): every NEW live game of ours (tapes written by tools/tape_sub.py in
the given live tape dirs) that we LOST, or that was against a player rated >= MINR, is added to data/tapes_spec in the
leader-tape format kagjax/train/tapes.py reads (the same conversion as make_specialist_tapes.py, which rebuilds the whole
bank from all live dirs - those are not all on this box, so here the bank is only ever appended to). The opponent's
seat becomes a league opponent for the anti-specialist RL; teams.txt is extended with its name.
Prints one line per added LOSS.   Usage: srv_spec_add.py <live tape dir> [...]      env MINR (2400)"""
import sys, os, json, gzip, glob
K = "/Users/asad.ali/kaggriculture"; OUT = f"{K}/data/tapes_spec"; MINR = float(os.environ.get("MINR", "2400"))
ip = f"{OUT}/index.json"; index = json.load(open(ip)); tf = f"{OUT}/teams.txt"
names = set(open(tf).read().split("|")) if os.path.exists(tf) else set()
added = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*.json.gz")):
        try:
            t = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        ep = int(t.get("ep", 0))
        if not ep or str(ep) in index or t.get("res") not in ("W", "L"):
            continue
        if not (t["res"] == "L" or (t.get("opp_rating") or 0) >= MINR):
            continue
        s = int(t["seat"]); nm = list(t.get("names") or ["?", "?"]); opp = nm[1 - s] or t.get("opp") or "?"
        if opp in ("?", "Syed Asad Ali"):
            continue
        teams = ["us:" + (nm[s] or "?"), opp] if s == 0 else [opp, "us:" + (nm[s] or "?")]
        out = dict(ep=ep, seed=int(t["seed"]), teams=teams, rewards=t.get("rewards"), statuses=["DONE", "DONE"], shops=[],
                   actions=t["actions"], day=t.get("end", "")[:10] or None)
        json.dump(out, gzip.open(f"{OUT}/{ep}.json.gz", "wt"), separators=(",", ":"))
        index[str(ep)] = dict(teams=teams, rewards=t.get("rewards"), statuses=["DONE", "DONE"], day=out["day"],
                              src=os.path.basename(d.rstrip("/")), res_vs_us=t["res"], opp_rating=t.get("opp_rating"))
        names.add(opp); added += 1
        if t["res"] == "L":
            rw = t.get("rewards") or [0, 0]
            print(f"SPEC BANK += LOSS ep {ep} vs {opp} ({(t.get('opp_rating') or 0):.0f}) margin {rw[s] - rw[1 - s]:+,.0f} "
                  f"[{os.path.basename(d.rstrip('/'))}]", flush=True)
if added:
    json.dump(index, open(ip, "w")); open(tf, "w").write("|".join(sorted(n for n in names if n)))
    print(f"specialist bank: +{added} games -> {len(index)}", flush=True)
