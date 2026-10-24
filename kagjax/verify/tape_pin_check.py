"""Are D's TOP tape worlds (kagjax/train/tapes.py + real_backend.step_tapes) the recorded worlds?

step_tapes feeds the tape seed's natural end-of-day words to the engine. The engine draws farm 0's weeds first, farm 1's
from word 2*E0 on, and the shop from word 2*(E0+E1) on (E = empty tiles at the day end), so as soon as the learner's
farm differs from the recorded player's, the leader's world changes: its weeds (when the leader is seat 1) and the
shop sequence (always). ppo_live / topruler pin both (set_forced_shops(recorded shops); set_weed_prefix(1, e0) when the
leader is seat 1). This script plays each tape with the leader replaying its recorded actions and the learner seat
played by PASS or a scripted bot, in the C++ engine (natural words == JAX step_tapes semantics, the JAX engine being
bit-exact with it) and with the reference pinning, and reports the shop sequences and the leader's banks.
usage (server, CPU): python tape_pin_check.py --n 30 --bot V94
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
os.environ.setdefault("KAG_ENGINE", os.path.join(HERE, "cov_build", "opt"))
sys.path.insert(0, f"{R}/tools")
sys.path.insert(0, f"{R}/kagjax/train")
sys.path.insert(0, f"{R}/kagjax")
import numpy as np  # noqa: E402
import harness as H  # noqa: E402
from harness import kagsim_trace as K, PASS  # noqa: E402
import topruler  # noqa: E402
import tapes as TP  # noqa: E402  (D's selection of leader seats; read only)


def play(d, leader, factory, pin, shops, e0):
    g = K.Game(int(d["seed"]), 720, False)
    if pin:
        g.set_forced_shops(shops)
        if leader == 1:
            g.set_weed_prefix(1, e0)
    us = 1 - leader
    ag = factory() if factory else None
    A = d["actions"][leader]
    for t in range(720):
        a_l = A[t] if t < len(A) and isinstance(A[t], dict) else dict(PASS)
        if ag is None:
            a_u = dict(PASS)
        else:
            try:
                a_u = ag(g.observe(us))
            except Exception:
                a_u = dict(PASS)
        now = [None, None]; now[leader] = a_l; now[us] = a_u
        g.step(now[0], now[1])
        if g.done:
            break
    return g.reward(leader), g.reward(us), g.shops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--bot", default="PASS")
    args = ap.parse_args()
    sel, _ = TP.select([f"{R}/data/tapes_top12"], TP.LEADERS_DEFAULT, "train", None)
    sel = sel[:: max(1, len(sel) // args.n)][:args.n]
    fac = None if args.bot == "PASS" else H.mk(args.bot)
    rows = []
    for p, leader in sel:
        d = json.load(gzip.open(p, "rt"))
        shops, e0, exact = topruler.recorded_world(p, d)
        if not exact:
            continue
        rec = d["rewards"][leader]
        nat = play(d, leader, fac, False, shops, e0)
        pin = play(d, leader, fac, True, shops, e0)
        rows.append(dict(ep=os.path.basename(p), leader_seat=leader, rec=rec, leader_nat=nat[0], leader_pin=pin[0],
                         us_nat=nat[1], us_pin=pin[1], shops_rec=list(shops), shops_nat=list(nat[2]),
                         shops_differ=list(nat[2]) != list(shops)))
        r = rows[-1]
        print(f"{r['ep']} leader seat {leader}: recorded {rec:,.0f} | natural words (JAX TOP) {nat[0]:,.0f} | pinned "
              f"(ppo_live) {pin[0]:,.0f} | shops differ from recording: {r['shops_differ']}", flush=True)
    a = np.array([[r["rec"], r["leader_nat"], r["leader_pin"], r["us_nat"], r["us_pin"]] for r in rows])
    sd = np.mean([r["shops_differ"] for r in rows])
    s1 = [r for r in rows if r["leader_seat"] == 1]
    print(f"RESULT learner={args.bot} n={len(rows)}: shop sequence differs from the recording in {sd:.0%} of natural-word "
          f"tape worlds | leader bank mean: recorded {a[:, 0].mean():,.0f}, natural {a[:, 1].mean():,.0f}, pinned "
          f"{a[:, 2].mean():,.0f} | natural - pinned {np.mean(a[:, 1] - a[:, 2]):+,.0f} (se "
          f"{np.std(a[:, 1] - a[:, 2]) / np.sqrt(len(a)):,.0f}); learner margin natural {np.mean(a[:, 3] - a[:, 1]):+,.0f} "
          f"vs pinned {np.mean(a[:, 4] - a[:, 2]):+,.0f} | leader seat 1: {len(s1)} tapes")
    json.dump(rows, open(os.path.join(HERE, "out", f"tape_pin_{args.bot}.json"), "w"))


if __name__ == "__main__":
    main()
