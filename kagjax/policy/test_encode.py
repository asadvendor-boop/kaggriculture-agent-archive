"""Parity of the JAX encoder with features6.encode_obs (the v6 encoder every shipped model uses).

1. constants equal to il/features.py / il/features6.py
2. lookup tables equal to the reference functions (price table: every entry; log table: sample + edges)
3. every "integer / constant" float feature: JAX float32 division == reference float64-then-float32, exhaustively
   over generous integer ranges
4. real games: replay recorded top-12 games (data/tapes_top12) in the C++ engine, both seats, every step; compare
   planes (uint8, exact), glob (float32, bitwise), pos, unit features (bitwise), valid rows.

run: kagjax/policy/run1.sh -m kagjax.policy.test_encode [n_games] [every]
"""
import sys, time, json, os
import numpy as np
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K, encode as E, refdata
import features as F
import features6 as F6

CHUNK = 256


def check_consts():
    assert K.OPS == F.OPS and K.N_OPS == F.N_OPS and K.N_QTY == F.N_QTY and K.MAX_UNITS == F.MAX_UNITS
    assert K.MARKET_HEADS == F.MARKET_HEADS and K.BIG == F.BIG
    assert K.ITEMS == F.ITEMS and K.PRODUCTS == F.PRODUCTS and K.CROPS == F.CROPS and K.ANIMALS == F.ANIMALS
    assert K.SHOPS == F.SHOPS and K.BASE == F.BASE and K.CROP_INFO == F.CROP_INFO and K.ANIMAL_INFO == F.ANIMAL_INFO
    assert set(K.SHED_ACCESS) == F.SHED_ACCESS and K.SPAWN == F.SPAWN and K.C == F.C and K.G5 == F.G and K.U == F.U
    assert K.G6 == F6.G and K.MKT == F6.MKT and K.SHOP_ITEMS == F6.SHOP_ITEMS and K.HORIZONS == F6.HORIZONS
    assert (K.I0, K.FLOOR, K.HINGE) == (F6.I0, F6.FLOOR, F6.HINGE)
    assert all(F.KIND[None if n == "EMPTY" else n] == i for i, n in enumerate(K.KIND_NAMES)) and len(F.KIND) == 6
    print("constants: equal to features.py / features6.py", flush=True)


def check_tables():
    t = K.tables()
    bad = 0
    for i, p in enumerate(K.PRODUCTS):
        ref = np.zeros(K.PRICE_N, np.float32)
        ref[:] = [F6.price(p, x) / F6.MKT[p][0] for x in range(K.PRICE_LO, K.PRICE_LO + K.PRICE_N)]
        bad += int((ref.view(np.int32) != t["price_ratio"][i].view(np.int32)).sum())
    assert bad == 0, bad
    rng = np.random.RandomState(0)
    ms = np.concatenate([np.arange(0, 5000), rng.randint(0, K.LOG_N, 50000), [K.LOG_N - 1]])
    g = np.zeros(1, np.float32); badl = 0
    for m in ms:
        g[0] = np.log1p(max(0.0, float(m))) / 12                  # the reference expression, scalar path
        badl += int(g.view(np.int32)[0] != t["log_money"][m:m + 1].view(np.int32)[0])
    assert badl == 0, badl
    print(f"tables: price table inv in [{K.PRICE_LO}, {K.PRICE_LO + K.PRICE_N}) x 9 exact; log1p(money)/12 table exact on {len(ms)} sampled entries", flush=True)


def check_divisions():
    """(a) informational: plain JAX float32 division by a constant (XLA rewrites x / c into x * (1/c));
    (b) the encoder's quotient tables through the jitted lookup == reference float64 division stored as float32,
    exhaustively over each table's domain (+ out-of-range clipping for the saturating features)."""
    a = np.arange(0, 800)
    plain = np.asarray(jax.jit(lambda x: x.astype(jnp.float32) / jnp.float32(720.0))(jnp.asarray(a)))
    ref = (a.astype(np.float64) / 720.0).astype(np.float32)
    print(f"divisions: plain XLA float32 step/720 differs from the reference in {int((plain != ref).sum())}/800 values "
          f"(x/c -> x*(1/c) rewrite); the encoder uses exact tables instead", flush=True)
    tabs = E.tables_jnp()
    tot = 0
    for name, (dv, lo, hi) in K.QUOT.items():
        n = np.arange(lo - 5, hi + 6, dtype=np.int32)
        got = np.asarray(jax.jit(lambda n, name=name: E._q(tabs, name, n))(jnp.asarray(n)))
        ref = (np.clip(n, lo, hi).astype(np.float64) / dv).astype(np.float32)
        assert (got.view(np.int32) == ref.view(np.int32)).all(), name
        tot += len(n)
    m = np.concatenate([np.arange(0, 1 << 16), np.random.RandomState(1).randint(0, K.LOG_N, 200000)]).astype(np.float32)
    got = np.asarray(jax.jit(lambda m: E._money_q(m, tabs))(jnp.asarray(m)))
    assert (got.view(np.int32) == (m.astype(np.float64) / 1e5).astype(np.float32).view(np.int32)).all()
    tot += len(m)
    d = np.arange(-70000, 70000, dtype=np.float32)
    f = jax.jit(lambda d: jnp.clip(tabs["mdiff_q"][jnp.clip(d, -K.MDIFF, K.MDIFF).astype(jnp.int32) + K.MDIFF], -3, 3))
    got = np.asarray(f(jnp.asarray(d)))
    assert (got.view(np.int32) == np.clip(d.astype(np.float64) / 2e4, -3, 3).astype(np.float32).view(np.int32)).all()
    tot += len(d)
    pr = np.arange(0, K.OBS_PRICE_N, dtype=np.int64)
    for i, p in enumerate(K.PRODUCTS):
        assert (tabs["obs_price_q"][i] == (pr.astype(np.float64) / K.BASE[p]).astype(np.float32)).all()
    print(f"divisions: {tot:,} table quotients bit-exact (jitted lookups, incl. clipped out-of-range inputs)", flush=True)


def ref_encode(obs, prev):
    pl, g, pos, uf = F6.encode_obs(obs, prev)
    n = len(pos)
    P = np.zeros((16, 2), np.int32); Uf = np.zeros((16, K.U), np.float32); v = np.zeros(16, bool)
    P[:n] = pos; Uf[:n] = uf; v[:n] = True
    return pl, g, P, Uf, v


def run_games(n_games, every, seats=(0, 1), start=0):
    tabs = E.tables_jnp()
    files = refdata.tape_files(n_games, start=start)
    stats = dict(games=0, obs=0, planes_bad=0, glob_bad=0, pos_bad=0, uf_bad=0, valid_bad=0, glob_bad_idx={},
                 max_units=0, steps=set())
    t_ref = t_raw = t_jax = 0.0
    for gi, f in enumerate(files):
        refs, raws, prevs = [], [], []
        for t, s, obs, prev in refdata.iter_tape_obs(f, every=every, seats=seats):
            t0 = time.time(); refs.append(ref_encode(obs, prev)); t_ref += time.time() - t0
            t0 = time.time(); raws.append(E.raw_from_obs(obs)); prevs.append(E.prev_array(prev)); t_raw += time.time() - t0
            stats["max_units"] = max(stats["max_units"], 1 + len(obs["farms"][s]["hands"]))
            stats["steps"].add(t)
        for c0 in range(0, len(raws), CHUNK):
            rr = raws[c0:c0 + CHUNK]; pp = prevs[c0:c0 + CHUNK]; n = len(rr)
            rr = rr + [rr[-1]] * (CHUNK - n); pp = pp + [pp[-1]] * (CHUNK - n)       # fixed shape: one compile
            t0 = time.time()
            out = jax.device_get(E.encode_batch(E.stack(rr), E.stack(pp), tabs))
            t_jax += time.time() - t0
            for j in range(n):
                pl, g, P, Uf, v = refs[c0 + j]
                stats["planes_bad"] += int((out["planes"][j] != pl).sum())
                gb = g.view(np.int32) != out["glob"][j].view(np.int32)
                stats["glob_bad"] += int(gb.sum())
                for i in np.nonzero(gb)[0]:
                    stats["glob_bad_idx"][int(i)] = stats["glob_bad_idx"].get(int(i), 0) + 1
                stats["pos_bad"] += int((out["pos"][j] != P).sum())
                stats["uf_bad"] += int((out["uf"][j].view(np.int32) != Uf.view(np.int32)).sum())
                stats["valid_bad"] += int((out["valid"][j] != v).sum())
        stats["games"] += 1; stats["obs"] += len(raws)
        print(f"  game {gi + 1}/{len(files)} {os.path.basename(f)}: {len(raws)} obs, cumulative mismatches planes "
              f"{stats['planes_bad']} glob {stats['glob_bad']} pos {stats['pos_bad']} uf {stats['uf_bad']} valid {stats['valid_bad']}",
              flush=True)
    stats["steps"] = len(stats["steps"])
    stats["time_ref_encode_ms_per_obs"] = 1e3 * t_ref / max(1, stats["obs"])
    stats["time_raw_from_obs_ms_per_obs"] = 1e3 * t_raw / max(1, stats["obs"])
    stats["time_jax_encode_ms_per_obs_incl_compile"] = 1e3 * t_jax / max(1, stats["obs"])
    return stats


def speed(b=64, reps=20):
    tabs = E.tables_jnp()
    f = refdata.tape_files(1)[0]
    raws, prevs = [], []
    for t, s, obs, prev in refdata.iter_tape_obs(f, every=11):
        raws.append(E.raw_from_obs(obs)); prevs.append(E.prev_array(prev))
        if len(raws) == b:
            break
    r, p = jax.device_put(E.stack(raws)), jax.device_put(E.stack(prevs))
    jax.block_until_ready(E.encode_batch(r, p, tabs))
    ts = []
    for _ in range(reps):
        t0 = time.time(); jax.block_until_ready(E.encode_batch(r, p, tabs)); ts.append(time.time() - t0)
    print(f"speed: jitted encode B={b}: median {np.median(ts)*1e3:.2f} ms ({np.median(ts)/b*1e3:.3f} ms/obs), 1 core", flush=True)
    return float(np.median(ts))


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    check_consts()
    check_tables()
    check_divisions()
    st = run_games(n_games, every, start=start)
    st["speed_B64_s"] = speed()
    print(json.dumps({k: v for k, v in st.items()}, indent=None, default=str))
    tag = "" if (n_games, every, start) == (24, 1, 0) else f"_{n_games}g_every{every}_start{start}"
    json.dump(st, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), f"results_encode{tag}.json"), "w"), indent=1, default=str)
    ok = st["planes_bad"] == st["glob_bad"] == st["pos_bad"] == st["uf_bad"] == st["valid_bad"] == 0
    print("ENCODER PARITY:", "EXACT" if ok else "MISMATCHES", f"on {st['obs']} observations from {st['games']} games")
