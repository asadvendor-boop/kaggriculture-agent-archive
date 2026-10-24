import sys, json, gzip, glob, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
mk = [collections.Counter(), collections.Counter()]; un = [collections.Counter(), collections.Counter()]
slots = collections.Counter(); nmk = [0, 0]; nun = [0, 0]; exact = 0; ng = 0
files = sorted(glob.glob("/Users/asad.ali/kaggriculture/data/tapes_st2/*.json.gz"))[:4] + sorted(glob.glob("/Users/asad.ali/kaggriculture/data/tapes_soh/*.json.gz"))[:4]
for f in files:
    d = json.load(gzip.open(f, "rt")); us = d["seat"]
    A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][s]] for s in (0, 1)]
    g = kagsim_trace.Game(d["seed"], 720, True); ng += 1
    for t in range(720):
        g.step(A[0][t], A[1][t])
        tr = g.trace_step(g.trace_len() - 1)
        for r in tr["receipts"]:
            who = 0 if r["seat"] == us else 1
            nmk[who] += 1
            slots[r["slot"]] += 1
            key = (r["op"], r["reason"], "full" if r["executed"] == r["req_n"] else ("none" if r["executed"] == 0 else "part"))
            mk[who][key] += 1
        for r in tr["unit_receipts"]:
            who = 0 if r["seat"] == us else 1
            nun[who] += 1
            if not r["executed"] and r["op"] != "PASS":
                un[who][r["op"]] += 1
        if g.done: break
    exact += abs(g.reward(0) - d["rewards"][0]) < .5 and abs(g.reward(1) - d["rewards"][1]) < .5
print("games", ng, "exact", exact, "max slot", max(slots))
for who, nm in ((0, "US"), (1, "OPP")):
    print(f"== {nm}: market orders {nmk[who]}, unit cmds {nun[who]}")
    for k, v in sorted(mk[who].items(), key=lambda kv: -kv[1])[:40]:
        if k[1] != "OK" or k[2] != "full":
            print("   mk", k, v)
    print("   unit not executed:", dict(un[who].most_common(20)))
