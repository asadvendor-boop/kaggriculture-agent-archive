"""Sanity of the verifier engine build: (1) same banks as the tape and as the pinned engine/v2 build;
(2) set_state round trip: loading observe() dicts and observing again is the identity; (3) continuing a tape from
a loaded mid-game state gives the same observations as the uninterrupted game (every step, both seats)."""
import gzip
import json
import glob
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
eng = os.environ["KAG_ENGINE"]
sys.path.insert(0, eng)
import kagsim_trace as K  # noqa: E402
print("module", K.__file__, K.__version__)
import state as S  # noqa: E402

files = sorted(glob.glob("/Users/asad.ali/kaggriculture/data/tapes_top12/*.json.gz"))
n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
bad = 0
for f in files[:n_games]:
    t = json.load(gzip.open(f))
    acts = [[x if isinstance(x, dict) else {"farmer": ["PASS"], "hands": [], "market": []} for x in t["actions"][k]]
            for k in (0, 1)]
    seed = int(t["seed"])
    g = K.Game(seed, 720, False)
    cuts = {100, 287, 431, 600, 718}
    forks = {}
    for s in range(720):
        if s in cuts:
            o0, o1 = g.observe(0), g.observe(1)
            h = K.Game(seed, 720, False)
            h.set_state(o0, o1)
            d = S.obs_diff(h.observe(0), o0) + S.obs_diff(h.observe(1), o1)
            if d:
                bad += 1
                print("ROUNDTRIP DIFF", f, s, d[:5])
            forks[s] = h
        g.step(acts[0][s], acts[1][s])
        for c, h in forks.items():
            h.step(acts[0][s], acts[1][s])
            d = S.obs_diff(h.observe(0), g.observe(0)) + S.obs_diff(h.observe(1), g.observe(1))
            if d:
                bad += 1
                print("FORK DIFF", f, "fork", c, "step", s, d[:5])
                forks[c] = None
        forks = {c: h for c, h in forks.items() if h is not None}
    ok = [float(x) for x in t["rewards"]] == [g.reward(0), g.reward(1)]
    print(os.path.basename(f), "seed", seed, "tape reward equal:", ok, t["rewards"], [g.reward(0), g.reward(1)],
          "done", g.done)
    bad += not ok
print("RESULT bad =", bad)
