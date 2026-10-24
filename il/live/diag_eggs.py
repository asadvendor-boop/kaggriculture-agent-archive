"""Sep 29: egg / goose census of our live games vs 2600+ (ALL + ALLG since 05:00 PKT), period A (before 14:30 PKT) vs B (evening):
geese bought (and by day 10), eggs sold (units, $), egg shops (BAKERY/BRUNCH_SPOT) in the town, fertilizer sold, cows bought."""
import sys, json, gzip, collections, statistics as St
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
K = "/Users/asad.ali/kaggriculture"; CUT = "2026-09-29T09:30"; DAY0 = "2026-09-29T00:00"
out = []
for sub, d in ((56651505, "tapes_all"), (56661802, "tapes_allg")):
    rows = {int(r["ep"]): r for r in json.load(open(f"{K}/data/rows_{sub}.json")) if r["res"] != "ERR"}
    for ep, r in rows.items():
        if (r.get("opp_rating") or 0) < 2600 or r["end"] < DAY0: continue
        try: t = json.load(gzip.open(f"{K}/data/{d}/{ep}.json.gz", "rt"))
        except Exception: continue
        us = t["seat"]; A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][s]] for s in (0, 1)]
        g = kagsim_trace.Game(int(t["seed"]), 720, True); c = [collections.Counter(), collections.Counter()]; shops = []
        for step in range(720):
            g.step(A[0][step], A[1][step]); tr = g.trace_step(g.trace_len() - 1)
            for rc in tr.get("receipts", []):
                if not rc.get("executed"): continue
                s = rc["seat"]; it = rc.get("item"); n = rc["executed"]
                if rc["op"] == "BUY_ANIMAL" and it == "GOOSE":
                    c[s]["geese"] += n
                    if step < 240: c[s]["geese_by_d10"] += n
                if rc["op"] == "BUY_ANIMAL" and it == "COW": c[s]["cows"] += n
                if rc["op"] == "SELL" and it == "EGG": c[s]["eggs"] += n; c[s]["egg$"] += rc["money_after"] - rc["money_before"]
                if rc["op"] == "SELL" and it == "FERTILIZER": c[s]["fert"] += n
            if g.done: break
        shops = g.observe(us)["town"]["unlocked_shops"]
        out.append(dict(per="B" if r["end"] >= CUT else "A", res=r["res"], eggshops=sum(1 for x in shops if x in ("BAKERY", "BRUNCH_SPOT")),
                        us=dict(c[us]), th=dict(c[1 - us]), opp=(r.get("opp") or "?").strip()))
json.dump(out, open(f"{K}/il/logs/diag_eggs.json", "w"))
for per in ("A", "B"):
    for es in ((0,), (1,), (2, 3, 4, 5, 6, 7, 8)):
        G = [x for x in out if x["per"] == per and x["eggshops"] in es]
        if not G: continue
        m = lambda who, k: St.mean(x[who].get(k, 0) for x in G)
        w = sum(x["res"] == "W" for x in G)
        print(f"{per} egg shops {'2+' if len(es) > 2 else es[0]}: n {len(G):2d} ({w}-{len(G)-w}) | geese us {m('us','geese'):4.1f} them {m('th','geese'):4.1f} "
              f"(by d10 {m('us','geese_by_d10'):.1f}/{m('th','geese_by_d10'):.1f}) | eggs sold {m('us','eggs'):5.0f}/{m('th','eggs'):5.0f} "
              f"(${m('us','egg$'):6,.0f}/${m('th','egg$'):6,.0f}) | fert sold {m('us','fert'):4.0f}/{m('th','fert'):4.0f} | cows {m('us','cows'):4.1f}/{m('th','cows'):4.1f}")
