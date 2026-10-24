"""TOP / SPEC gate (the auditor's re-check after the Sep 26 pinning fix, run again Sep 27 before enabling tape opponents).

Same comparison as kagjax/verify/tape_pin_check.py + tape_pin_jax.py, but with the TRAINER's code path on the JAX side:
for N leader seats of data/tapes_top12 and M specialist seats of data/tapes_spec (training split, evenly spaced,
recordings that reproduce exactly), the leader replays its recorded actions and the learner seat is played by
  (a) PASS, (b) the scripted bot V94 (reactive: its moves depend on the world it sees).
C++ (tools/harness engine): each world UNPINNED (the tape seed's natural words = the pre-fix JAX TOP) and PINNED (the
ppo_live / topruler reference: forced recorded shops + weed prefix when the leader is seat 1), V94's actions recorded.
JAX: tapes.bank_from_selection + RealBackend.step_tapes (exactly what the trainer runs), the learner seat replaying the
actions the C++ PINNED run produced.
GATE: JAX banks == C++ pinned banks (both seats, every world, both learners); and the unpinned C++ worlds differ from the
pinned ones in most worlds (the test has teeth).
usage: python kagjax/train/recheck_tape_pin.py [n_top=60] [n_spec=40]      -> kagjax/logs_server/D/tape_recheck.json
"""
import os, sys, json, gzip, time, copy
HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import tapes as TP
import state as AS
from test_tape_pin import jax_worlds

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
OUT = os.environ.get("RECHECK_OUT", f"{R}/kagjax/logs_server/D/tape_recheck.json")


def play_cpp(d, leader, bot, pin, shops, e0):
    """C++ tape world; returns (final money [2], the learner seat's actions [720])."""
    import harness as H
    g = H.kagsim_trace.Game(int(d["seed"]), 720, False)
    if pin:
        g.set_forced_shops(shops)
        if leader == 1:
            g.set_weed_prefix(1, e0)
    us = 1 - leader
    ag = None if bot == "PASS" else H.mk(bot)()
    A_ = d["actions"][leader]
    rec = []
    for t in range(720):
        a_l = A_[t] if t < len(A_) and isinstance(A_[t], dict) else dict(PASS)
        if ag is None:
            a_u = dict(PASS)
        else:
            try:
                a_u = ag(copy.deepcopy(g.observe(us)))
            except Exception:
                a_u = dict(PASS)
        rec.append(copy.deepcopy(a_u))
        now = [None, None]; now[leader] = a_l; now[us] = a_u
        g.step(now[0], now[1])
        if g.done:
            break
    rec += [dict(PASS)] * (720 - len(rec))
    return np.array([g.reward(0), g.reward(1)], np.float64), rec


def main():
    n_top = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_spec = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    import topruler
    t0 = time.time()
    sel_t, _ = TP.select([f"{R}/data/tapes_top12"], TP.TOP_TEAMS_DEFAULT, "train", None)
    sel_s, _ = TP.select([f"{R}/data/tapes_spec"], TP.TOP_TEAMS_DEFAULT, "train", None, any_opponent=True)
    pick = lambda sel, n: [sel[i] for i in np.linspace(0, len(sel) - 1, min(n, len(sel))).round().astype(int)]
    sel = pick(sel_t, n_top) + pick(sel_s, n_spec)
    spec = [False] * min(n_top, len(sel_t)) + [True] * min(n_spec, len(sel_s))
    bank = TP.bank_from_selection(sel, spec, both=True)       # (drops recordings that do not reproduce exactly)
    paths, T = bank["paths"], len(bank["lseat"])
    print(f"{T} tape seats ({int((~bank['spec']).sum())} leaders, {int(bank['spec'].sum())} specialists; "
          f"{bank['n_not_exact']} not reproducing skipped); leader in seat 1: {int((1 - bank['lseat']).sum())}", flush=True)
    res = dict(n=T, n_leaders=int((~bank["spec"]).sum()), n_specialists=int(bank["spec"].sum()), learners={})
    ok_all = True
    for bot in ("PASS", "V94"):
        pin_m, nat_m, acts = [], [], []
        for i, p in enumerate(paths):
            d = json.load(gzip.open(p, "rt"))
            leader = 1 - int(bank["lseat"][i])
            shops, e0, exact = topruler.recorded_world(p, d)
            m_pin, a_pin = play_cpp(d, leader, bot, True, shops, e0)
            m_nat, _ = play_cpp(d, leader, bot, False, shops, e0)
            pin_m.append(m_pin); nat_m.append(m_nat); acts.append(a_pin)
        pin_m, nat_m = np.stack(pin_m), np.stack(nat_m)
        b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bank.items()}
        for i in range(T):                                   # the learner seat replays the C++ pinned run's actions
            for t in range(720):
                uop, uarg, un, mt, mi, mn = AS.encode_action(acts[i][t])
                b2["luop"][i, t], b2["luarg"][i, t], b2["lun"][i, t] = uop, uarg, np.clip(un, -30000, 30000)
                b2["lmtype"][i, t], b2["lmitem"][i, t], b2["lmn"][i, t] = mt, mi, mn
        got = jax_worlds(b2, pinned=True).astype(np.float64)
        same = np.all(got == pin_m, axis=1)
        lead = 1 - bank["lseat"]
        teeth = np.array([nat_m[i, lead[i]] != pin_m[i, lead[i]] for i in range(T)])
        spf = bank["spec"]
        r = dict(jax_eq_cpp_pinned=int(same.sum()), jax_eq_cpp_pinned_leaders=int(same[~spf].sum()),
                 jax_eq_cpp_pinned_specialists=int(same[spf].sum()), unpinned_leader_bank_differs=int(teeth.sum()),
                 leader_bank_unpinned_minus_pinned=float(np.mean([nat_m[i, lead[i]] - pin_m[i, lead[i]] for i in range(T)])),
                 learner_margin_pinned=float(np.mean([pin_m[i, 1 - lead[i]] - pin_m[i, lead[i]] for i in range(T)])),
                 mismatches=[dict(ep=int(bank["meta"][i][0]), seat=int(lead[i]), cpp=pin_m[i].tolist(), jax=got[i].tolist())
                             for i in np.flatnonzero(~same)[:10]])
        res["learners"][bot] = r
        ok = bool(same.all()) and bool(teeth.mean() > 0.5)
        ok_all &= ok
        print(f"[{bot} learner] JAX trainer path == C++ pinned in {same.sum()}/{T} worlds (leaders {same[~spf].sum()}/{(~spf).sum()}, "
              f"specialists {same[spf].sum()}/{spf.sum()}); unpinned leader bank differs in {teeth.sum()}/{T} (mean "
              f"{r['leader_bank_unpinned_minus_pinned']:+,.0f}); learner margin (pinned) {r['learner_margin_pinned']:+,.0f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res["PASS"] = ok_all
    res["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("TOP/SPEC GATE PASS" if ok_all else "TOP/SPEC GATE FAIL", f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
