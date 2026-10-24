"""TOP tape worlds must be the recorded worlds even when the learner is NOT the recorded player (audit, Sep 26).

For N leader seats of the training bank, the learner seat is played by (a) PASS, (b) the recorded actions of a
DIFFERENT tape's same seat (a foreign, deterministic player).  Reference: the C++ engine with the ppo_live / topruler
pinning (set_forced_shops(recorded shops); set_weed_prefix(1, e0) when the leader is seat 1).  JAX: A's engine through
RealBackend.step_tapes (the trainer's code path; the learner seat's actions injected as the bank's l* arrays).
Both seats' final money must be identical in every world; also reports how far the UNPINNED (natural words) worlds
are from the reference, to show the test has teeth.
run: kagjax/train/run1.sh kagjax/train/test_tape_pin.py [n]      (server CPU: srv_launch.sh ... python ...)
"""
import os, sys, json, gzip, time, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import tapes as TP
import real_backend as RB
import actions as A
import state as AS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def cpp_world(path, leader, learner_actions):
    """Final money [2] of the C++ pinned tape world with the given learner-seat actions (list of 720 dicts)."""
    from harness import kagsim_trace
    import topruler
    d = json.load(gzip.open(path, "rt"))
    shops, e0, exact = topruler.recorded_world(path, d)
    g = kagsim_trace.Game(int(d["seed"]), 720, False); g.set_forced_shops(shops)
    if leader == 1:
        g.set_weed_prefix(1, e0)
    acts = d["actions"][leader]
    for t in range(720):
        now = [None, None]
        a = acts[t] if t < len(acts) and isinstance(acts[t], dict) else PASS
        now[leader] = a; now[1 - leader] = learner_actions[t]
        g.step(now[0], now[1])
        if g.done:
            break
    return np.array([g.reward(0), g.reward(1)])


def jax_worlds(bank, pinned=True):
    be = RB.RealBackend()
    cs = be.consts()
    T = len(bank["lseat"])
    dev = {k: jnp.asarray(v) for k, v in TP.device_arrays(bank, pinned).items()}
    st = be.reset(jax.random.PRNGKey(0), T)
    z = jnp.zeros((T, 16), jnp.int32); zo = jnp.zeros((T, 10), jnp.int32)
    dummy = A.Act(z, z, jnp.ones((T, 16), jnp.int32), jnp.zeros(T, jnp.int32), zo, zo, zo, jnp.zeros(T, jnp.int32))
    tid = jnp.arange(T); lead = 1 - jnp.asarray(bank["lseat"])

    @jax.jit
    def run(st, dev, cs):
        def body(s, k):
            return be.step_tapes(s, dummy, dummy, tid, lead, dev, k, cs), None
        s, _ = jax.lax.scan(body, st, jax.random.split(jax.random.PRNGKey(1), 720))
        return s.g.money

    return np.asarray(run(st, dev, cs))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    bank = TP.build_bank(split="train", max_n=n, both=True)
    sel, _ = TP.select([f"{TP.R}/data/tapes_top12"], os.environ.get("TOP_TEAMS", TP.LEADERS_DEFAULT), "train", n)
    # the same selection order as build_bank (it drops non-reproducing tapes; keep the matching ones)
    paths = [p for p, s in sel]
    eps = [m[0] for m in bank["meta"]]
    paths = [next(p for p in paths if int(os.path.basename(p).split(".")[0]) == e) for e in eps]
    T = len(eps)
    print(f"{T} leader seats; leader in seat 1: {int((1 - bank['lseat']).sum())}", flush=True)
    ok = True
    for mode in ("pass", "foreign"):
        b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bank.items()}
        learner = []
        for i in range(T):
            if mode == "pass":
                acts = [PASS] * 720
                for k in ("luop", "luarg", "lmtype", "lmitem", "lmn"):
                    b2[k][i] = 0
                b2["lun"][i] = 1
            else:                                    # the same seat of another recorded game: a foreign player
                j = (i + 1) % T
                dj = json.load(gzip.open(paths[j], "rt"))
                acts = [a if isinstance(a, dict) else PASS for a in dj["actions"][int(bank["lseat"][i])]][:720]
                acts += [PASS] * (720 - len(acts))
                for t in range(720):
                    uop, uarg, un, mt, mi, mn = AS.encode_action(acts[t])
                    b2["luop"][i, t], b2["luarg"][i, t], b2["lun"][i, t] = uop, uarg, np.clip(un, -30000, 30000)
                    b2["lmtype"][i, t], b2["lmitem"][i, t], b2["lmn"][i, t] = mt, mi, mn
            learner.append(acts)
        ref = np.stack([cpp_world(paths[i], 1 - int(bank["lseat"][i]), learner[i]) for i in range(T)])
        got = jax_worlds(b2, pinned=True)
        nat = jax_worlds(b2, pinned=False)
        same = np.all(got == ref, axis=1)
        same_nat = np.all(nat == ref, axis=1)
        lead = 1 - bank["lseat"]
        dlead = np.array([nat[i, lead[i]] - ref[i, lead[i]] for i in range(T)])
        print(f"[{mode} learner] pinned JAX == C++ pinned in {same.sum()}/{T} worlds; unpinned (natural words) in "
              f"{same_nat.sum()}/{T} (leader bank unpinned - pinned: mean {dlead.mean():+,.0f})", flush=True)
        for i in np.flatnonzero(~same)[:5]:
            print(f"   ep {eps[i]} leader seat {lead[i]}: C++ {ref[i].tolist()} JAX {got[i].tolist()}")
        ok &= bool(same.all())
    print(f"({time.time() - t0:.0f}s)")
    print("TAPE PIN OK" if ok else "TAPE PIN MISMATCH", flush=True)


if __name__ == "__main__":
    main()
