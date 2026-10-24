import sys, json, gzip
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
f = "/Users/asad.ali/kaggriculture/data/tapes_st2/114804416.json.gz"
d = json.load(gzip.open(f, "rt")); A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][s]] for s in (0, 1)]
g = kagsim_trace.Game(d["seed"], 720, True)
for t in range(720):
    g.step(A[0][t], A[1][t])
    if t in (0, 1, 300, 301):
        tr = g.trace_step(g.trace_len() - 1)
        print("step", t, "keys", list(tr.keys()))
        for k, v in tr.items():
            print("   ", k, json.dumps(v)[:600])
    if g.done: break
print("rewards", g.reward(0), g.reward(1), d["rewards"])
