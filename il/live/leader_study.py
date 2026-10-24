"""Per-team study of recorded play: (1) how much of each team's market play our output menu represents and why not,
(2) strategy profile from exact replays (revenue by product, animals/land/hires/wheat trading, final bank).
Usage: leader_study.py [n_games_per_team]"""
import sys, json, gzip, collections, multiprocessing as mp
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
R = "/Users/asad.ali/kaggriculture"
TEAMS = ["DSM", "M & M & P & Q", "DECEM", "Vadim Vasilenko", "Majkel1337", "Fourth Quadrant", "Unknown Mother-Goose",
         "tetsuya & yuanzhe & guoqi", "Kaggledew Valley 🏆", "mtmr_s1", "吃白饭的大肥鱼", "THIRD FARM CLUB", "SpaTaro",
         "Artem The Farmer 🍅", "Azat Akhtyamov", "Boey", "Snorlax"]
PROD = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
QMAX = {"SELL": 16, "BUY_SEED": 7, "BUY_ANIMAL": 3, "BUY_PRODUCT:WHEAT": 16, "BUY_PRODUCT:FERTILIZER": 3}


def study(item):
    import features as F
    from harness import kagsim_trace, PASS
    path, s = item
    d = json.load(gzip.open(path, "rt")); A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][k]] for k in (0, 1)]
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    rep = collections.Counter(); why = collections.Counter(); prof = collections.Counter(); first_cow = None
    for t in range(720):
        a = A[s][t]; m = a.get("market") or []
        if m:
            rep["turns"] += 1
            raw = [o for o in m[:10]]
            bad = [o for o in raw if not (isinstance(o, list) and o and o[0] in ("SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "HIRE", "BUY_LAND"))]
            good = [o for o in raw if o not in bad]
            key = collections.Counter((o[0], o[1] if len(o) > 1 else "") for o in good if o[0] != "HIRE")
            over = []
            for o in good:
                q = o[2] if len(o) > 2 else 1
                try: q = int(q)
                except Exception: q = -1
                k = o[0] if o[0] != "BUY_PRODUCT" else f"BUY_PRODUCT:{o[1] if len(o) > 1 else ''}"
                if (k in QMAX and q > QMAX[k] and not (o[0] == "SELL" and q >= 1000) and not (k == "BUY_PRODUCT:WHEAT" and q == 35)) or q <= 0:
                    over.append(k)
            hires = sum(1 for o in good if o[0] == "HIRE")
            reasons = []
            if bad: reasons.append("empty/dead padding order")
            if any(v > 1 for v in key.values()): reasons.append("repeated same-item order")
            if over: reasons.append("quantity beyond menu (" + ",".join(sorted(set(over))) + ")")
            if hires > 10: reasons.append("more than 10 hires")
            if reasons:
                rep["unrep"] += 1; why.update(reasons)
        g.step(A[0][t], A[1][t])
        for rc in g.trace_step(g.trace_len() - 1).get("receipts", []):
            if rc["seat"] != s or not rc.get("executed"): continue
            dm = (rc.get("money_after") or 0) - (rc.get("money_before") or 0); op = rc.get("op"); it = rc.get("item")
            if op == "SELL": prof["rev:" + str(it)] += dm; prof["sold:" + str(it)] += rc["executed"]
            elif op == "BUY_ANIMAL":
                prof["animal:" + str(it)] += rc["executed"]
                if it == "COW" and first_cow is None: first_cow = t // 24
            elif op == "BUY_LAND": prof["land"] += 1
            elif op == "HIRE": prof["hire_orders"] += rc["executed"]
            elif op == "BUY_PRODUCT" and it == "WHEAT": prof["wheat_bought"] += rc["executed"]
        if g.done: break
    prof["bank"] = g.reward(s); prof["opp_bank"] = g.reward(1 - s); prof["first_cow_day"] = first_cow if first_cow is not None else -1
    return rep, why, prof


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    games = {t: [] for t in TEAMS}
    for d_ in ("data/tapes_pull", "data/tapes_daily"):
        idx = json.load(open(f"{R}/{d_}/index.json"))
        for ep, m in sorted(idx.items(), key=lambda kv: -int(kv[0])):          # most recent first
            tt = m.get("teams") or []
            for s in (0, 1):
                if s < len(tt) and tt[s] in games and len(games[tt[s]]) < n:
                    games[tt[s]].append((f"{R}/{d_}/{ep}.json.gz", s))
    with mp.get_context("spawn").Pool(4) as pool:
        print(f"{'team':26s} n  | market turns: unrep | top reasons                                   | bank  | rev share: WHE STR MEL MIL WOO FER TOM EGG CAR | cows sheep geese 1st-cow-day land wheat-bought/sold")
        for team in TEAMS:
            if not games[team]: print(f"{team:26s} no games"); continue
            res = pool.map(study, games[team]); k = len(res)
            rep = sum((r[0] for r in res), collections.Counter()); why = sum((r[1] for r in res), collections.Counter())
            P = [r[2] for r in res]; tot = np.mean([sum(p[f"rev:{x}"] for x in PROD) for p in P])
            share = " ".join(f"{np.mean([p['rev:'+x] for p in P])/max(1,tot)*100:3.0f}" for x in ("WHEAT", "STRAWBERRY", "MELON", "MILK", "WOOL", "FERTILIZER", "TOMATO", "EGG", "CARROT"))
            fc = [p["first_cow_day"] for p in P if p["first_cow_day"] >= 0]
            print(f"{team[:26]:26s} {k:2d} | {rep['unrep']/max(1,rep['turns']):5.1%} | {', '.join(f'{w} {c}' for w, c in why.most_common(2))[:46]:46s} | {np.mean([p['bank'] for p in P])/1e3:4.0f}k | {share} | "
                  f"{np.mean([p['animal:COW'] for p in P]):4.1f} {np.mean([p['animal:SHEEP'] for p in P]):4.1f} {np.mean([p['animal:GOOSE'] for p in P]):4.1f} {np.median(fc) if fc else -1:4.0f} {np.mean([p['land'] for p in P]):4.1f} {np.mean([p['wheat_bought'] for p in P]):4.0f}/{np.mean([p['sold:WHEAT'] for p in P]):4.0f}", flush=True)
