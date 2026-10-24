"""Sep 29: paired comparison restricted to worlds where the recorded opponent stayed intact (kept >= 95% of its recorded score) in BOTH
runs - tape-collapse artifacts removed. Works for top_bench JSONs (rows carry rec_opp) and 464 ladder JSONs (rec opp read from the tape).
Usage: intact_cmp.py <base.json> <cand.json> [...]"""
import sys, json, gzip, statistics as St
W = {w["ep"]: w for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json"))}
_rc = {}
def rec_opp(r):
    if r.get("rec_opp") is not None: return r["rec_opp"]
    ep = r["ep"]
    if ep not in _rc:
        w = W[ep]; t = json.load(gzip.open(w["path"], "rt")); rw = t.get("rewards"); _rc[ep] = rw[1 - w["seat"]] if rw else None
    return _rc[ep]
def load(fn): return {(r["ep"], r.get("seat", 0), r.get("opp")): r for r in json.load(open(fn))}
b = load(sys.argv[1])
for fn in sys.argv[2:]:
    c = load(fn); k = [x for x in b if x in c]
    ok = [x for x in k if rec_opp(b[x]) and b[x]["opp_bank"] >= 0.95 * rec_opp(b[x]) and c[x]["opp_bank"] >= 0.95 * rec_opp(c[x])]
    d = [c[x]["margin"] - b[x]["margin"] for x in ok]; hi = [x for x in ok if (b[x].get("opp_rating") or 0) >= 2400]; dh = [c[x]["margin"] - b[x]["margin"] for x in hi]
    print(f"{fn.split('/')[-1][:58]:58s} intact {len(ok)}/{len(k)} | wins {sum(b[x]['margin']>0 for x in ok)} -> {sum(c[x]['margin']>0 for x in ok)} | "
          f"diff {St.mean(d):+,.0f} (se {St.pstdev(d)/len(d)**0.5:,.0f}) | 2400+ ({len(hi)}): {sum(b[x]['margin']>0 for x in hi)} -> {sum(c[x]['margin']>0 for x in hi)} {St.mean(dh) if dh else 0:+,.0f}")
