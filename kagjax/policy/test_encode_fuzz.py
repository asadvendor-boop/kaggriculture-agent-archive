"""Encoder fuzz: real observations perturbed into unusual ranges, JAX encoder vs features6.encode_obs (bit-exact).

Perturbations (each observation gets a random subset): extra hands up to 30 units with random positions and
inventories (plane counts over ALL units, glob inventory sums over all units, unit features for 16), money up to
2^20 (table range) and a rival far richer / poorer, market inventories anywhere in [0, 32767], prices up to 4095,
duplicate shops, hires_today up to 30, any quadrant count, tile fields (yield 0..20, planted / placed day in the
future or far past, fertilized_until, max_lifespan, dryness, care bonus, random booleans), random step, random prev
ops with gaps.  Structural validity is kept (kinds / crops / animals are real values).

run: kagjax/policy/run1.sh -m kagjax.policy.test_encode_fuzz [n_obs]
"""
import sys, copy, json, os
import numpy as np
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K, encode as E, refdata
from kagjax.policy.test_encode import ref_encode


def perturb(o, rng):
    o = copy.deepcopy(o)
    me = int(o["player"])
    if rng.rand() < 0.5:
        o["step"] = int(rng.randint(0, 720))
    for f in (0, 1):
        fm = o["farms"][f]
        if rng.rand() < 0.4:                                        # many units
            n = int(rng.randint(1, 31))
            fm["hands"] = [[int(rng.randint(0, 10)), int(rng.randint(0, 10))] for _ in range(n - 1)]
            fm["farmer"] = [int(rng.randint(0, 10)), int(rng.randint(0, 10))]
        if rng.rand() < 0.5:
            fm["money"] = float(rng.randint(0, K.LOG_N))
        if rng.rand() < 0.3:
            fm["hires_today"] = int(rng.randint(0, 31))
        if rng.rand() < 0.3:
            fm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"][:int(rng.randint(1, 5))]
        for row in fm["tiles"]:
            for x, t in enumerate(row):
                if not isinstance(t, dict) or rng.rand() > 0.3:
                    continue
                if t.get("kind") == "PLANT":
                    t["yield_units"] = int(rng.randint(0, 21)); t["planted_day"] = int(rng.randint(-5, 40))
                    t["watered_today"] = bool(rng.rand() < 0.5); t["consecutive_unwatered"] = int(rng.randint(0, 4))
                    t["fertilized_until_day"] = int(rng.randint(-1, 35)); t["max_lifespan_step"] = int(rng.randint(-1, 800))
                    t["crop"] = K.CROPS[rng.randint(0, 5)]
                elif "animal" in t:
                    t["yield_units"] = int(rng.randint(0, 21)); t["placed_day"] = int(rng.randint(-5, 40))
                    for b in ("fed_today", "cared_today", "fertilizer_available"):
                        t[b] = bool(rng.rand() < 0.5)
                    t["consecutive_unfed"] = int(rng.randint(0, 4)); t["pending_care_bonus"] = int(rng.randint(0, 6))
                    t["animal"] = K.ANIMALS[rng.randint(0, 3)]
    priv = o["private"]
    n_own = 1 + len(o["farms"][me]["hands"])
    invs = list(priv.get("inventories") or [])
    while len(invs) < n_own:
        invs.append({})
    invs = invs[:n_own]
    for u in range(n_own):
        if rng.rand() < 0.3:
            invs[u] = {K.ITEMS[i]: int(rng.randint(0, 40)) for i in rng.choice(12, rng.randint(1, 5), replace=False)}
    priv["inventories"] = invs
    if rng.rand() < 0.5:
        priv["shed"] = {it: int(rng.randint(0, 120)) for it in K.ITEMS}
    if rng.rand() < 0.5:
        priv["seeds"] = {c: int(rng.randint(0, 40)) for c in K.CROPS}
    if rng.rand() < 0.5:
        o["market"]["inventory"] = {p: int(rng.randint(0, K.PRICE_N)) for p in K.PRODUCTS}
    if rng.rand() < 0.5:
        o["market"]["prices"] = {p: int(rng.randint(1, K.OBS_PRICE_N)) for p in K.PRODUCTS}
    if rng.rand() < 0.4:
        o["town"]["unlocked_shops"] = [K.SHOPS[i] for i in rng.randint(0, 8, rng.randint(0, 12))]
    prev = None
    if rng.rand() < 0.8:
        prev = [int(v) for v in rng.randint(-1, K.N_OPS, rng.randint(0, 17))]
    return o, prev


def main(n_obs=6000, chunk=256):
    rng = np.random.RandomState(7)
    base = []
    for f in refdata.tape_files(6, start=17):
        for t, s, o, prev in refdata.iter_tape_obs(f, every=17):
            base.append(o)
    tabs = E.tables_jnp()
    st = dict(n=0, planes_bad=0, glob_bad=0, pos_bad=0, uf_bad=0, valid_bad=0, max_units=0, glob_bad_idx={})
    for c0 in range(0, n_obs, chunk):
        obs, prevs, refs = [], [], []
        for _ in range(chunk):
            o, prev = perturb(base[rng.randint(len(base))], rng)
            obs.append(o); prevs.append(prev); refs.append(ref_encode(o, prev))
            st["max_units"] = max(st["max_units"], 1 + len(o["farms"][int(o["player"])]["hands"]))
        out = jax.device_get(E.encode_batch(E.stack([E.raw_from_obs(o) for o in obs]),
                                            E.stack([E.prev_array(p) for p in prevs]), tabs))
        for j in range(chunk):
            pl, g, P, Uf, v = refs[j]
            st["planes_bad"] += int((out["planes"][j] != pl).sum())
            gb = g.view(np.int32) != out["glob"][j].view(np.int32)
            st["glob_bad"] += int(gb.sum())
            for i in np.nonzero(gb)[0]:
                st["glob_bad_idx"][int(i)] = st["glob_bad_idx"].get(int(i), 0) + 1
            st["pos_bad"] += int((out["pos"][j] != P).sum())
            st["uf_bad"] += int((out["uf"][j].view(np.int32) != Uf.view(np.int32)).sum())
            st["valid_bad"] += int((out["valid"][j] != v).sum())
        st["n"] += chunk
    print(f"fuzz: {st['n']} perturbed observations (up to {st['max_units']} units): mismatches planes {st['planes_bad']} "
          f"glob {st['glob_bad']} {st['glob_bad_idx']} pos {st['pos_bad']} uf {st['uf_bad']} valid {st['valid_bad']}", flush=True)
    json.dump(st, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), "results_encode_fuzz.json"), "w"), indent=1)
    ok = st["planes_bad"] == st["glob_bad"] == st["pos_bad"] == st["uf_bad"] == st["valid_bad"] == 0
    print("ENCODER FUZZ:", "EXACT" if ok else "MISMATCHES")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
