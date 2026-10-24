"""Sep 29 evening slide diagnosis: our live games vs 2600+ (ALL + ALLG), exact C++ replay of both seats, cash flow per channel
(money_after - money_before of every executed receipt: SELL:<item>, BUY_PRODUCT:<item>, BUY_SEED, BUY_ANIMAL:<item>, HIRE, BUY_LAND),
escapes, our wheat drain-straddle P&L; period A (before 09:30 UTC = 14:30 PKT) vs B (after). Prints where the us-minus-them gap moved."""
import sys, json, gzip, collections, statistics as St
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
K = "/Users/asad.ali/kaggriculture"; CUT = "2026-09-29T09:30"; DAY0 = "2026-09-29T00:00"
games = []
for sub, d in ((56651505, "tapes_all"), (56661802, "tapes_allg")):
    rows = {int(r["ep"]): r for r in json.load(open(f"{K}/data/rows_{sub}.json")) if r["res"] != "ERR"}
    for ep, r in rows.items():
        if (r.get("opp_rating") or 0) < 2600 or r["end"] < DAY0: continue
        try: t = json.load(gzip.open(f"{K}/data/{d}/{ep}.json.gz", "rt"))
        except Exception: continue
        us = t["seat"]; A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][s]] for s in (0, 1)]
        g = kagsim_trace.Game(int(t["seed"]), 720, True); cf = [collections.Counter(), collections.Counter()]; esc = [0, 0]; strad = 0.0
        pend = {}
        for step in range(720):
            g.step(A[0][step], A[1][step]); tr = g.trace_step(g.trace_len() - 1)
            for rc in tr.get("receipts", []):
                if not rc.get("executed"): continue
                s = rc["seat"]; dm = rc["money_after"] - rc["money_before"]; op = rc["op"]; it = rc.get("item") or ""
                key = f"{op}:{it}" if op in ("SELL", "BUY_PRODUCT", "BUY_ANIMAL") else op
                cf[s][key] += dm
                if s == us and it == "WHEAT":
                    if op == "BUY_PRODUCT" and step % 4 == 0: pend[step] = pend.get(step, 0) + dm
                    if op == "SELL" and (step - 1) in pend: strad += dm + pend.pop(step - 1)
            for e in tr.get("escapes", []) or []:
                if isinstance(e, dict) and e.get("seat") in (0, 1): esc[e["seat"]] += 1
            if g.done: break
        games.append(dict(ep=ep, per="B" if r["end"] >= CUT else "A", res=r["res"], margin=r["margin"], opp=(r.get("opp") or "?").strip(),
                          orat=r["opp_rating"], sub=r.get("opp_sub"), us=cf[us], th=cf[1 - us], esc=(esc[us], esc[1 - us]), strad=strad))
print(f"games vs 2600+ since Sep 29 05:00 PKT: {len(games)}")
keys = sorted(set().union(*[set(g_["us"]) | set(g_["th"]) for g_ in games]))
for per in ("A", "B"):
    G = [x for x in games if x["per"] == per]; w = sum(x["res"] == "W" for x in G)
    print(f"\nPERIOD {per} ({'05:00-14:30' if per == 'A' else '14:30-now'} PKT): {w}-{len(G)-w}, mean margin {St.mean(x['margin'] for x in G):+,.0f} | "
          f"escapes us {St.mean(x['esc'][0] for x in G):.2f} them {St.mean(x['esc'][1] for x in G):.2f} | our straddle P&L {St.mean(x['strad'] for x in G):+,.0f}/game")
gap = {}
for k in keys:
    a = [x["us"][k] - x["th"][k] for x in games if x["per"] == "A"]; b = [x["us"][k] - x["th"][k] for x in games if x["per"] == "B"]
    gap[k] = (St.mean(a), St.mean(b))
print("\nUS MINUS THEM cash per game by channel (A -> B, change), biggest moves first:")
for k, (a, b) in sorted(gap.items(), key=lambda kv: (kv[1][1] - kv[1][0]))[:10]:
    print(f"  {k:22s} {a:+9,.0f} -> {b:+9,.0f}  ({b - a:+,.0f})")
print("\nperiod B losses by opponent:")
L = collections.Counter(x["opp"] for x in games if x["per"] == "B" and x["res"] != "W")
W_ = collections.Counter(x["opp"] for x in games if x["per"] == "B" and x["res"] == "W")
for o, n in L.most_common(12): print(f"  {o[:24]:24s} lost {n}  won {W_.get(o, 0)}")
json.dump([dict(x, us=dict(x["us"]), th=dict(x["th"])) for x in games], open(f"{K}/il/logs/diag_slide.json", "w"))
