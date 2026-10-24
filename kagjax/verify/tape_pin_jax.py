"""The JAX side of tape_pin_check.py: what D's step_tapes does (A's engine, the tape seed's natural words, the
leader seat replaying its recorded actions via state.encode_action, the learner seat PASS) on the first tapes of
out/tape_pin_PASS.json; the leader's final bank must equal the C++ 'natural' number there (and differ from 'pinned').
usage (server, CPU): python tape_pin_jax.py 6
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, f"{R}/kagjax")
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
import rng as RNG  # noqa: E402
import state as S  # noqa: E402

rows = json.load(open(os.path.join(HERE, "out", "tape_pin_PASS.json")))[:int(sys.argv[1])]
B = len(rows)
tapes = [json.load(gzip.open(f"{R}/data/tapes_top12/{r['ep']}", "rt")) for r in rows]
words = np.moveaxis(RNG.batch_words([int(d["seed"]) for d in tapes]), 0, 1)       # [B,30,624]
st = E.init(B)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
for t in range(720):
    acts = [[PASS, PASS] for _ in range(B)]
    for b, (r, d) in enumerate(zip(rows, tapes)):
        L = r["leader_seat"]
        a = d["actions"][L][t] if t < len(d["actions"][L]) else None
        acts[b][L] = a if isinstance(a, dict) else PASS
    st = E.step(st, S.encode_actions([a[0] for a in acts]), S.encode_actions([a[1] for a in acts]), words)
m = np.asarray(st.g.money)
for b, r in enumerate(rows):
    L = r["leader_seat"]
    print(f"{r['ep']} leader seat {L}: JAX (step_tapes semantics) leader bank {m[b, L]:,} | C++ natural {r['leader_nat']:,.0f}"
          f" | C++ pinned (ppo_live) {r['leader_pin']:,.0f} | recorded {r['rec']:,.0f}")
