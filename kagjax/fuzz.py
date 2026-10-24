"""Adversarial full-engine fuzz: real bots (tools/harness AGENTS, read-only) or random agents play in the C++
reference; their actions are randomly perturbed (random / malformed unit ops, extra hand actions for units that do
not exist, PLANT over-demand bursts, DROP / PICKUP / PLACE spam, random and malformed market orders, forced
same-slot same-item SELL/BUY interactions between the two players, junk action dicts) and the same actions are fed
to the JAX engine; the full state is compared after every step (as in parity.py).

usage: env1.sh fuzz.py [--games 8] [--batch 8] [--seed 0] [--p 0.25] [--bots V94,M7,...] [--random 0.25]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import harness as H  # noqa: E402
from harness import kagsim_trace, PASS  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

import consts as C  # noqa: E402
import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402

DEFAULT_BOTS = ["V94", "V48", "E604", "M7", "STACK1", "V57", "V54", "HWME", "SALES12", "Kimi621b", "FORK_M"]
ITEMS = C.ITEM_NAMES
JUNK_COUNTS = [0, -1, -5, 1, 2, 3, 5, 10, 30, 99, 100, 101, 1000, 10 ** 6, 10 ** 7, 10 ** 30, "3", "7", " 4 ",
               "x", 2.7, 5.9, -0.5, True, False, None, [2], "1_000"]


def rand_unit(r):
    u = r.random()
    if u < 0.04:
        return r.choice([[], ["FOO"], "NORTH", None, 5, ["PLANT"], ["PICKUP"], ["PLACE"], ["PICKUP", 5],
                         ["PLACE", None, 2], ["INVALID"], ("NORTH",), ["DROP", "WHEAT", 5]])
    op = r.choice(C.UNIT_OPS + ["NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP", "PLANT", "HARVEST"])
    if op in ("PICKUP", "PLACE"):
        a = [op, r.choice(ITEMS + ["FOO", "WHEAT", "MILK"])]
        if r.random() < 0.8:
            a.append(r.choice(JUNK_COUNTS) if r.random() < 0.2 else r.randint(1, 60))
        return a
    if op == "PLANT":
        return [op, r.choice(ITEMS[:5] + ["EGG", "FOO"])]
    return [op]


def rand_order(r, obs=None):
    u = r.random()
    if u < 0.06:
        return r.choice([[], ["SELL"], ["SELL", "WHEAT"], ["SELL", "WHEAT", 0], ["SELL", "WHEAT", -3],
                         ["SELL", "FOO", 3], ["SELL", 3, 3], "HIRE", ["HIRE", "x", 9], None, ["BUY_LAND", 1],
                         ["NONE", "WHEAT", 3], ["BUY_SEED", "WHEAT", "2"], ["BUY_SEED", "WHEAT", 2.9],
                         ["BUY_SEED", "WHEAT", True], ["BUY_PRODUCT", "MELON", 3], ["BUY_ANIMAL", "EGG", 1],
                         ["SELL", "GOOSE", 1], ["BUY_SEED", "WHEAT", 10 ** 30], ["SELL", "WHEAT", "abc"]])
    if u < 0.14:
        return ["HIRE"]
    if u < 0.15:
        return ["BUY_LAND"]
    if u < 0.3:
        return ["BUY_SEED", r.choice(ITEMS[:5]), r.choice([1, 1, 2, 3, 5, 10]) if r.random() < 0.97 else
                r.choice([25, 100, 10 ** 6])]
    if u < 0.37:
        return ["BUY_ANIMAL", r.choice(["GOOSE", "COW", "SHEEP"]), r.choice([1, 1, 1, 2]) if r.random() < 0.97 else
                r.choice([3, 10, 200])]
    if u < 0.55:
        return ["BUY_PRODUCT", r.choice(["WHEAT", "WHEAT", "FERTILIZER", "FERTILIZER", "MILK"]),
                r.choice([1, 2, 3, 5, 10]) if r.random() < 0.9 else r.choice([40, 100, 150, 10 ** 6])]
    it = r.choice(ITEMS[:9])
    if obs is not None and r.random() < 0.6:
        shed = obs["private"]["shed"]
        have = [k for k, v in shed.items() if v > 0 and k in ITEMS[:9]]
        if have:
            it = r.choice(have)
    return ["SELL", it, r.choice([1, 2, 3, 5, 10, 40, 100, 10 ** 6])]


def random_agent(r, obs):
    f = obs["farms"][obs["player"]]
    n = 1 + len(f["hands"])
    hands = [rand_unit(r) for _ in range(max(0, n - 1 + r.choice([0, 0, 0, 1, 2, -1])))]
    return {"farmer": rand_unit(r), "hands": hands, "market": [rand_order(r, obs) for _ in range(r.randint(0, 12))]}


def perturb(r, a, obs, p):
    if not isinstance(a, dict):
        a = dict(PASS)
    a = {"farmer": a.get("farmer", ["PASS"]), "hands": list(a.get("hands", []) or []),
         "market": list(a.get("market", []) or []) if isinstance(a.get("market", []), list) else a.get("market")}
    if r.random() < p * 0.4:                                   # random unit ops
        for _ in range(r.randint(1, 4)):
            k = r.randint(0, len(a["hands"]))
            if k == 0:
                a["farmer"] = rand_unit(r)
            else:
                a["hands"][k - 1] = rand_unit(r)
    if r.random() < p * 0.15:                                  # hand actions for units that do not exist
        a["hands"] += [rand_unit(r) for _ in range(r.randint(1, 4))]
    if r.random() < p * 0.15:                                  # PLANT burst (atomic over-demand check)
        crop = r.choice(ITEMS[:5])
        k = r.randint(1, 4)
        a["farmer"] = ["PLANT", crop]
        for i in range(min(k, len(a["hands"]))):
            a["hands"][i] = ["PLANT", crop]
    if r.random() < p * 0.2:                                   # shed traffic
        for i in range(r.randint(1, 3)):
            u = r.choice([["DROP"], ["PICKUP", r.choice(ITEMS), r.choice([1, 5, 50, 1000])],
                          ["PLACE", r.choice(ITEMS), r.choice([1, 5, 50, 1000])]])
            k = r.randint(0, len(a["hands"]))
            if k == 0:
                a["farmer"] = u
            else:
                a["hands"][k - 1] = u
    if isinstance(a["market"], list) and r.random() < p * 0.6:  # random orders anywhere in the list
        for _ in range(r.randint(1, 5)):
            a["market"].insert(r.randint(0, len(a["market"])), rand_order(r, obs))
    if r.random() < p * 0.01:
        a = r.choice([None, 7, "PASS", {"farmer": "NORTH"}, {"market": "SELL"}, {"hands": [], "market": None},
                      {"farmer": ["EAST"], "market": [["HIRE"]] * 12}, {}])
    return a


def entangle(r, a0, a1, p):
    """Force same-slot same-item market interactions between the players."""
    if not (isinstance(a0, dict) and isinstance(a1, dict)):
        return
    m0, m1 = a0.get("market"), a1.get("market")
    if not (isinstance(m0, list) and isinstance(m1, list)) or r.random() > p:
        return
    for i in range(min(len(m0), C.NSLOT)):
        o = m0[i]
        if isinstance(o, list) and len(o) >= 3 and o[0] in ("SELL", "BUY_PRODUCT") and isinstance(o[1], str):
            if r.random() < 0.7:
                it = o[1] if r.random() < 0.7 else r.choice(["WHEAT", "FERTILIZER"])
                new = [r.choice(["SELL", "BUY_PRODUCT"]), it, r.choice([1, 3, 10, 50, 100, 10 ** 6])]
                while len(m1) <= i:
                    m1.append(rand_order(r))
                m1[i] = new


def run_batch(r, B, bots, p, p_random, verbose=True, cfg=None):
    seeds = [r.randrange(2 ** 31) for _ in range(B)]
    refs = [kagsim_trace.Game(s, 720, False) for s in seeds]
    kinds, agents = [], []
    for b in range(B):
        pair = []
        for seat in range(2):
            if r.random() < p_random:
                pair.append(("random", None))
            else:
                name = r.choice(bots)
                pair.append((name, H.mk(name)()))
        kinds.append([k for k, _ in pair])
        agents.append([ag for _, ag in pair])
    st = E.init(B)
    words = R.batch_words(seeds)
    obs = [[refs[b].observe(0), refs[b].observe(1)] for b in range(B)]
    bad = [None] * B
    errs = 0
    t0 = time.time()
    for s in range(720):
        acts = []
        for b in range(B):
            pair = []
            for seat in range(2):
                if agents[b][seat] is None:
                    a = random_agent(r, obs[b][seat])
                else:
                    try:
                        a = agents[b][seat](obs[b][seat])
                    except Exception:
                        errs += 1
                        a = dict(PASS)
                pair.append(perturb(r, a, obs[b][seat], p))
            entangle(r, pair[0], pair[1], p)
            if r.random() < 0.5:
                entangle(r, pair[1], pair[0], p)
            acts.append(pair)
        st = E.step(st, S.encode_actions([acts[b][0] for b in range(B)]),
                    S.encode_actions([acts[b][1] for b in range(B)]), words[min(s // 24, 29)], cfg=cfg)
        g = jax.tree_util.tree_map(np.asarray, st.g)
        t = int(st.t)
        for b in range(B):
            refs[b].step(acts[b][0], acts[b][1])
            obs[b] = [refs[b].observe(0), refs[b].observe(1)]
            if bad[b] is None:
                for seat in (0, 1):
                    d = S.obs_diff(S.game_to_obs(g, b, seat, t), obs[b][seat])
                    if d:
                        bad[b] = (s, seat, d[:8], acts[b])
                        if verbose:
                            print(f"  MISMATCH seed {seeds[b]} {kinds[b]} after step {s} seat {seat}: {d[:6]}")
                            print(f"     actions: {acts[b]}")
                        break
    g = jax.tree_util.tree_map(np.asarray, st.g)
    res = []
    for b in range(B):
        res.append({"seed": seeds[b], "agents": kinds[b], "ok": bad[b] is None,
                    "first_bad_step": None if bad[b] is None else bad[b][0],
                    "reward": [int(g.money[b, 0]), int(g.money[b, 1])],
                    "ref": [refs[b].reward(0), refs[b].reward(1)], "oob": bool(g.oob[b])})
    if verbose:
        print(f"  batch {B} games {time.time() - t0:.0f}s, bot exceptions {errs}")
    return res


def pick_cfg(args):
    name = "par" if args.par else args.cfg
    return {"seq": E.CFG_SEQ, "par": E.CFG_PAR, "gpu": E.CFG_GPU, "unrolled": E.CFG_UNROLLED,
            "default": None}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--random", type=float, default=0.2, help="probability a seat is a random agent")
    ap.add_argument("--bots", default=",".join(DEFAULT_BOTS))
    ap.add_argument("--par", action="store_true", help="same as --cfg par")
    ap.add_argument("--cfg", default="seq", choices=["seq", "par", "gpu", "unrolled", "default"],
                    help="loop structure: engine.CFG_SEQ / CFG_PAR / CFG_GPU (identical results)")
    args = ap.parse_args()
    r = random.Random(args.seed)
    bots = [b for b in args.bots.split(",") if os.path.exists(H.AGENTS.get(b, b))]
    allres = []
    for i in range(0, args.games, args.batch):
        res = run_batch(r, min(args.batch, args.games - i), bots, args.p, args.random,
                        cfg=pick_cfg(args))
        allres += res
        for x in res:
            print(("OK  " if x["ok"] else "BAD ") + str(x), flush=True)
    n_ok = sum(x["ok"] and x["reward"] == x["ref"] for x in allres)
    print(f"RESULT {n_ok}/{len(allres)} fuzz games exact through all steps")


if __name__ == "__main__":
    main()
