"""Land timing and sheep count of our live neural games, losses vs wins (for the third-field guard / sheep brake rules).
Per game (our seat, exact C++ replay): day of each BUY_LAND, cash at the start of days 8-14, sheep bought by day, the
first day where 2 lands were owned and cash >= $4,000 + reserve while the 3rd was not yet bought (a rule trigger).
Usage: land_sheep_diag.py <tape_dir> [<tape_dir> ...]      env NP (3), RESERVE (500)"""
import sys, os, json, gzip, glob, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
RESERVE = int(os.environ.get("RESERVE", "500"))


def one(item):
    from harness import kagsim_trace, PASS
    path, = item
    try:
        d = json.load(gzip.open(path, "rt")); s = int(d["seat"])
        A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][k]] for k in (0, 1)]
        g = kagsim_trace.Game(int(d["seed"]), 720, True)
    except Exception:
        return None
    lands = []; cash = {}; sheep = collections.Counter(); anim = collections.Counter(); trig = None; nq = None; qn0 = None
    for t in range(720):
        o = g.observe(s); f = o["farms"][s]; day = t // 24
        if t % 24 == 0:
            cash[day] = int(f["money"])
        if qn0 is None:
            qn0 = list(f.get("unlocked_quadrants") or [])
        nl = len(lands)
        if trig is None and nl == 2 and f["money"] >= 4000 + RESERVE:
            trig = (day, t % 24, int(f["money"]))
        g.step(A[0][t], A[1][t]); tr = g.trace_step(g.trace_len() - 1)
        for rc in tr.get("receipts", []):
            if rc.get("seat") != s or not rc.get("executed"):
                continue
            if rc.get("op") == "BUY_LAND":
                lands.append(day)
            elif rc.get("op") == "BUY_ANIMAL":
                anim[rc.get("item")] += rc["executed"]
                if rc.get("item") == "SHEEP":
                    sheep[day] += rc["executed"]
        if g.done:
            break
    sheep_before3 = sum(n for dd, n in sheep.items() if len(lands) < 3 or dd <= lands[2])
    return dict(ep=d["ep"], res=d.get("res"), opp=d.get("opp"), orat=d.get("opp_rating") or 0, margin=d["rewards"][s] - d["rewards"][1 - s],
                lands=lands, cash=cash, sheep=dict(sheep), sheep_tot=sum(sheep.values()), sheep_before3=sheep_before3, anim=dict(anim), trig=trig, q0=qn0)


if __name__ == "__main__":
    files = [f for d_ in sys.argv[1:] for f in sorted(glob.glob(f"{d_}/*.json.gz"))]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "3"))) as pool:
        R = [r for r in pool.map(one, [(f,) for f in files], chunksize=2) if r and r["res"] in ("W", "L")]
    json.dump(R, open(os.environ.get("OUT", "/tmp/land_sheep.json"), "w"))
    for lab, G in (("LOSSES", [r for r in R if r["res"] == "L"]), ("WINS", [r for r in R if r["res"] == "W"]), ("WINS vs 2400+", [r for r in R if r["res"] == "W" and r["orat"] >= 2400])):
        n = len(G)
        if not n:
            continue
        has3 = [r for r in G if len(r["lands"]) >= 3]
        no3 = [r for r in G if len(r["lands"]) < 3]
        no3_trig = [r for r in no3 if r["trig"]]
        print(f"\n== {lab}: n {n} | 3rd land bought {len(has3)} ({len(has3)/n:.0%}), day of 3rd: {collections.Counter(r['lands'][2] for r in has3).most_common(6)}")
        print(f"   land days most common: {collections.Counter(tuple(r['lands']) for r in G).most_common(5)}")
        print(f"   no 3rd land: {len(no3)}; of those, cash >= ${4000+RESERVE} with 2 lands at some point: {len(no3_trig)}"
              f" (first day {collections.Counter(r['trig'][0] for r in no3_trig).most_common(6)})")
        print(f"   sheep total mean {np.mean([r['sheep_tot'] for r in G]):.1f}, >12 before 3rd land: {sum(r['sheep_before3'] > 12 for r in G)}"
              f" | cash start of day 9/10/11/12 (mean): " + " / ".join(f"{np.mean([r['cash'].get(dd, 0) for r in G]):,.0f}" for dd in (9, 10, 11, 12)))
    print("\nLOSSES in detail (margin, opponent, lands, first trigger (day,hour,cash), sheep by day, animals):")
    for r in sorted([r for r in R if r["res"] == "L"], key=lambda r: r["margin"]):
        print(f"  {r['margin']:+8,.0f} {str(r['opp'])[:20]:20s} {r['orat']:5.0f} lands {str(r['lands']):12s} trig {str(r['trig']):20s} sheep {r['sheep_tot']:2d} {dict(sorted(r['sheep'].items()))} {r['anim']}")
