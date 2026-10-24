"""Adversarial equivalence test: parallel (round-based) unit phase vs the sequential per-unit loop.
Random farms (plants of every crop and age, structures with / without animals, weeds, locked quadrants), units
crowded on few tiles (incl. the 4 shed-access tiles), inventories with random insertion stamps, sheds near the
100 cap, and random unit actions biased to same-tile conflicts (HARVEST+PLANT, DIG+BUILD+PLACE animal, FEED+CARE,
PLACE vs shed path, PICKUP/DROP/PLACE chains). Every output array must be identical.
usage: env1.sh test_units.py [--n 20000]
"""
import argparse
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import consts as C
import engine as E


def rand_farm(r):
    tiles = np.tile(C.FRESH_TILE, (C.NT, 1)).astype(np.int32)
    nq = r.integers(1, 5)
    for i in range(C.NT):
        q = C.QUAD_OF_TILE[i]
        if q >= nq:
            tiles[i, C.F_KIND] = C.K_LOCKED
            continue
        u = r.random()
        day = 10
        if u < 0.3:
            continue
        if u < 0.4:
            tiles[i, C.F_KIND] = C.K_WEED
        elif u < 0.7:
            c = r.integers(0, 5)
            tiles[i, [C.F_KIND, C.F_WHAT, C.F_DAY, C.F_WATERED, C.F_DRY, C.F_YIELD]] = [
                C.K_PLANT, c, day - r.integers(0, 12), r.integers(0, 2), r.integers(0, 2), r.integers(0, 7)]
            tiles[i, C.F_MLS] = r.choice([-1, 240, 300])
            tiles[i, C.F_FERTU] = r.choice([-1, day, day + 1])
        else:
            k = r.choice([C.K_COOP, C.K_PASTURE])
            tiles[i, C.F_KIND] = k
            if r.random() < 0.6:
                a = C.GOOSE if k == C.K_COOP else r.choice([C.COW, C.SHEEP])
                tiles[i, [C.F_WHAT, C.F_ANIMAL, C.F_DAY, C.F_YIELD, C.F_FED, C.F_CARED, C.F_FERTAV, C.F_CARE]] = [
                    a, 1, day - r.integers(0, 9), r.integers(0, 5), r.integers(0, 2), r.integers(0, 2),
                    r.integers(0, 2), r.integers(0, 3)]
    for i in (44, 45, 54, 55):                    # shed-access tiles as empty structures (PLACE animal vs shed path)
        if tiles[i, C.F_KIND] != C.K_LOCKED and r.random() < 0.4:
            tiles[i] = C.FRESH_TILE
            tiles[i, C.F_KIND] = r.choice([C.K_COOP, C.K_PASTURE])
    n_units = int(r.choice([1, 3, 8, 15, 25, 40]))
    hot = [r.integers(0, C.NT) for _ in range(3)] + [44, 45, 54, 55]
    ux = np.zeros(C.U, np.int32)
    uy = np.zeros(C.U, np.int32)
    for u in range(C.U):
        p = int(r.choice(hot)) if r.random() < 0.7 else int(r.integers(0, C.NT))
        ux[u], uy[u] = p % 10, p // 10
    shed = np.zeros(C.N_ITEMS, np.int32)
    tot = int(r.choice([0, 10, 60, 95, 99, 100]))
    for _ in range(tot):
        shed[r.choice([0, 0, 8, 9, 10, 11, int(r.integers(0, 12))])] += 1
    seeds = r.integers(0, 4, C.N_CROPS).astype(np.int32)
    inv = np.zeros((C.U, C.N_ITEMS), np.int32)
    ins = np.zeros((C.U, C.N_ITEMS), np.int32)
    for u in range(C.U):
        if r.random() < 0.5:
            for it in r.choice(C.N_ITEMS, size=r.integers(1, 5), replace=False):
                inv[u, it] = r.integers(1, 30)
            if r.random() < 0.5:
                inv[u, r.integers(9, 12)] = r.integers(1, 4)
            ins[u] = r.permutation(C.N_ITEMS) - 50
    return tiles, ux, uy, shed, seeds, inv, ins, np.int32(n_units)


def rand_actions(r):
    ops = [C.OP_PASS, C.OP_NORTH, C.OP_EAST, C.OP_PICKUP, C.OP_DROP, C.OP_PLACE, C.OP_PLANT, C.OP_WATER,
           C.OP_HARVEST, C.OP_FERTILIZE, C.OP_DIG, C.OP_BUILD_COOP, C.OP_BUILD_PASTURE, C.OP_FEED,
           C.OP_COLLECT_FERTILIZER, C.OP_CARE, C.OP_INVALID]
    uop = r.choice(ops, C.U).astype(np.int32)
    uarg = r.choice(list(range(12)) + [9, 10, 11, 0, 8, 255], C.U).astype(np.int32)
    uarg = np.where((uop == C.OP_PLANT) & (r.random(C.U) < 0.85), uarg % 5, uarg).astype(np.int32)
    un = r.choice([-1, 0, 1, 2, 5, 30, 1000], C.U).astype(np.int32)
    pl = r.random(C.U) < 0.15                    # extra PLACE-animal traffic
    uop = np.where(pl, C.OP_PLACE, uop).astype(np.int32)
    uarg = np.where(pl, r.integers(9, 12, C.U), uarg).astype(np.int32)
    return uop, uarg, un


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = np.random.default_rng(args.seed)

    @jax.jit
    def both(tiles, ux, uy, shed, seeds, inv, ins, n_units, uop, uarg, un, day, t):
        seq = jax.vmap(lambda *a: E._unit_phase(*a, umax=C.U), in_axes=(0,) * 11 + (None, None))
        par = jax.vmap(lambda *a: E._unit_phase_par(*a, rmax=C.U, kmax=C.U), in_axes=(0,) * 11 + (None, None))
        args_ = (tiles, ux, uy, shed, seeds, inv, ins, uop, uarg, un, n_units, day, t)
        return seq(*args_), par(*args_)

    bad = 0
    t0 = time.time()
    for c0 in range(0, args.n, args.chunk):
        n = min(args.chunk, args.n - c0)
        farms = [rand_farm(r) for _ in range(n)]
        acts = [rand_actions(r) for _ in range(n)]
        cols = [np.stack([f[k] for f in farms]) for k in range(8)]
        a = [np.stack([x[k] for x in acts]) for k in range(3)]
        s_out, p_out = both(*cols, *a, jnp.int32(10), jnp.int32(250))
        names = ["tiles", "ux", "uy", "shed", "seeds", "inv", "ins"]
        for k, name in enumerate(names):
            x = np.asarray(s_out[k])
            y = np.asarray(p_out[k])
            if name == "ins":                          # stamps only matter where the item is held
                held = np.asarray(s_out[5]) > 0
                x = np.where(held, x, 0)
                y = np.where(held, y, 0)
            if name in ("ux", "uy"):                   # only live units
                live = np.arange(C.U)[None, :] < cols[7][:, None]
                x = np.where(live, x, 0)
                y = np.where(live, y, 0)
            diff = np.nonzero(np.any((x != y).reshape(n, -1), axis=1))[0]
            if len(diff):
                bad += len(diff)
                b = diff[0]
                print(f"MISMATCH {name} in {len(diff)} farms; e.g. farm {c0 + b}")
                if bad > 50:
                    break
        print(f"{c0 + n} farms, {bad} mismatching, {time.time() - t0:.1f}s", flush=True)
    print("RESULT", "PASS" if bad == 0 else "FAIL", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
