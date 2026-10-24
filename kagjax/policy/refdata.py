"""Reference data for the parity tests: replay recorded games (data/tapes_top12) in the reference C++ engine and
yield real observations (+ the previous-step op labels of the recorded actions, the way live_rollout's scripted
opening builds `prev`)."""
import sys, json, gzip, glob, os
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, PASS          # noqa: E402

TAPES = "/Users/asad.ali/kaggriculture/data/tapes_top12"


def tape_files(n, skip_seed0=True, stride=37, start=0):
    """n tape paths spread over the directory (deterministic), skipping tapes without a real seed."""
    fs = sorted(glob.glob(f"{TAPES}/*.json.gz"))
    out = []
    i = start
    while len(out) < n and i < start + stride * len(fs):
        f = fs[i % len(fs)]; i += stride
        if f in out:
            continue
        if skip_seed0:
            d = json.load(gzip.open(f, "rt"))
            if int(d.get("seed") or 0) == 0:
                continue
        out.append(f)
    return out


def _act(d, s, t):
    a = d["actions"][s][t] if t < len(d["actions"][s]) else None
    return a if isinstance(a, dict) else dict(PASS)


def iter_tape_obs(path, every=1, seats=(0, 1), max_steps=720):
    """Yields (t, seat, obs, prev_ops) along the recorded game replayed in the engine."""
    import features as F
    d = json.load(gzip.open(path, "rt"))
    g = kagsim_trace.Game(int(d["seed"]), 720, False)
    prev = [None, None]
    for t in range(max_steps):
        obs = [g.observe(0), g.observe(1)]
        if t % every == 0:
            for s in seats:
                yield t, s, obs[s], prev[s]
        acts = [_act(d, 0, t), _act(d, 1, t)]
        for s in (0, 1):
            a = acts[s]
            prev[s] = [F.unit_op_label(u)[0] for u in [a.get("farmer")] + list(a.get("hands") or [])]
        g.step(acts[0], acts[1])
        if g.done:
            break
