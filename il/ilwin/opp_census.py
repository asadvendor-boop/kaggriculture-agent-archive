import json, gzip, glob, collections
D = "/Users/asad.ali/kaggriculture/data"
voc = json.load(open("/Users/asad.ali/kaggriculture/il/data/team_vocab.json")); V = set(voc)
lad = {int(w["ep"]) for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json"))}
ll = {int(w["ep"]) for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/top_worlds_llall_300.json"))}
tot = collections.Counter()
for d in ["tapes_st2", "tapes_soh", "tapes_n20mw8", "tapes_npse", "tapes_nps", "tapes_n20mw5", "tapes_neural"]:
    c = collections.Counter(); names = collections.Counter(); rb = collections.Counter()
    for f in sorted(glob.glob(f"{D}/{d}/*.json.gz")):
        x = json.load(gzip.open(f, "rt")); us = x["seat"]; rw = x["rewards"]
        if rw[0] is None or rw[1] is None: continue
        m = rw[us] - rw[1 - us]; r = x.get("opp_rating") or 0; ep = int(x["ep"])
        key = "W" if m > 0 else "L"
        c[key] += 1
        if ep in lad: c["in464"] += 1; continue
        if ep in ll: c["inLLALL_" + key] += 1
        for th in (2400, 2500, 2600):
            if r >= th: c[f"{key}>={th}"] += 1
        if key == "L":
            names[x["opp"]] += 1
            c["L_opp_in_vocab"] += x["opp"] in V
        if abs(m) < 500: c[key + "_tie500"] += 1
    print(d, dict(c))
    print("   loss opponents:", names.most_common(12))
