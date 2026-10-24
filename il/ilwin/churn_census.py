import json, gzip, glob, collections, sys
D = "/Users/asad.ali/kaggriculture/data"
def q(o):
    try: return max(0, int(o[2]))
    except Exception: return 0
for d in ["tapes_st2", "tapes_soh", "tapes_n20mw8", "tapes_npse", "tapes_nps", "tapes_n20mw5"]:
    stats = collections.Counter(); per = []
    for f in sorted(glob.glob(f"{D}/{d}/*.json.gz")):
        x = json.load(gzip.open(f, "rt")); us = x["seat"]; rw = x["rewards"]
        if rw[0] is None or rw[1] is None: continue
        won = rw[us] > rw[1 - us]; w = us if won else 1 - us
        A = x["actions"][w]
        buys = pairs = wash = big = 0; bw = 0
        for t in range(719):
            a = A[t] if isinstance(A[t], dict) else {}
            mk = [o for o in (a.get("market") or [])[:10] if isinstance(o, list) and len(o) >= 3]
            b = sum(q(o) for o in mk if o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
            s = sum(q(o) for o in mk if o[0] == "SELL" and o[1] == "WHEAT")
            bw += b
            if b >= 10: big += 1
            if b > 0 and s > 0: wash += 1
            if t % 4 == 0 and b >= 10:
                a2 = A[t + 1] if isinstance(A[t + 1], dict) else {}
                s2 = sum(q(o) for o in (a2.get("market") or [])[:10] if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] == "WHEAT")
                if s2 >= 0.5 * b: pairs += 1
        role = "us" if won else "opp"
        stats[role + "_games"] += 1; stats[role + "_bigbuy_steps"] += big; stats[role + "_pairs"] += pairs; stats[role + "_wash_steps"] += wash
        stats[role + "_wheat_bought"] += bw
        if pairs >= 5: stats[role + "_straddler_games"] += 1
    print(d, dict(stats))
