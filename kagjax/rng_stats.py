"""Training-mode randomness check: do i.i.d. jax.random day words give the same game distribution as the exact
MT19937 words of real seeds?

Both seats play the same DETERMINISTIC scripted bot (wheat farmer written in JAX below: hires 3 hands every
morning, walks 2-row bands in a snake, digs weeds, plants / waters / harvests wheat, sells all wheat, buys seeds,
buys land when rich), so a game is a deterministic function of its world randomness only. Two arms of N games:
  exact : N random Kaggle-style seeds (uniform in [0, 2^31)), words = rng.day_words(seed, day)  (== C++ / Kaggle)
  train : words drawn by the engine from jax.random keys (engine.init(B, seed) + step(words=None))
Compared (two-sample tests): final money per seat, weeds spawned per game and per day, empty tiles rolled per day,
the pooled weed rate (vs 0.005), shop draws per unlock index (chi-square homogeneity) and the (1st, 2nd) shop pair.
usage: python rng_stats.py --n 4096 --out logs_server/rng_stats.json
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy import stats

import consts as C
import engine as E
import rng as R
from state import Actions

i32 = jnp.int32


def wheat_bot(g, t, p):
    """Deterministic scripted wheat farmer for seat p (reads only its own farm). Returns Actions [B,...]."""
    B = g.money.shape[0]
    hour = t % 24
    day = t // 24
    tiles = g.tiles[:, p]
    ux = g.ux[:, p]
    uy = g.uy[:, p]
    n = g.n_units[:, p]
    uidx = jnp.arange(C.U)
    live = uidx[None, :] < n[:, None]
    pos = uy * 10 + ux
    tk = lambda f: jnp.take_along_axis(tiles[..., f], pos, axis=1)
    kind, pday, watered = tk(C.F_KIND), tk(C.F_DAY), tk(C.F_WATERED) == 1
    y0 = (2 * (uidx % 5))[None, :]
    y1 = y0 + 1
    s = jnp.where(uy == y0, ux, 10 + (9 - ux))
    sn = (s + 1) % 20
    tx = jnp.where(sn < 10, sn, 9 - (sn - 10))
    ty = jnp.where(sn < 10, y0, y1)
    mv = jnp.where(uy < y0, C.OP_SOUTH, jnp.where(uy > y1, C.OP_NORTH, jnp.where(
        tx > ux, C.OP_EAST, jnp.where(tx < ux, C.OP_WEST, jnp.where(ty > uy, C.OP_SOUTH, C.OP_NORTH)))))
    want_plant = live & (kind == C.K_EMPTY)
    rank = jnp.cumsum(want_plant.astype(i32), axis=1) - want_plant.astype(i32)
    can_plant = want_plant & (rank < g.seeds[:, p, C.WHEAT][:, None])        # never trip the atomic seed check
    op = jnp.where(kind == C.K_WEED, C.OP_DIG,
         jnp.where(can_plant, C.OP_PLANT,
         jnp.where((kind == C.K_PLANT) & (day - pday >= 4), C.OP_HARVEST,
         jnp.where((kind == C.K_PLANT) & ~watered, C.OP_WATER, mv))))
    op = jnp.where(kind == C.K_LOCKED, mv, op)
    op = jnp.where(live, op, C.OP_PASS).astype(i32)
    uarg = jnp.zeros((B, C.U), i32)                                          # WHEAT
    un = jnp.ones((B, C.U), i32)
    money = g.money[:, p]
    seeds = g.seeds[:, p, C.WHEAT]
    z = jnp.zeros((B,), i32)
    mt = jnp.zeros((B, C.NSLOT), i32)
    mi = jnp.zeros((B, C.NSLOT), i32)
    mn = jnp.zeros((B, C.NSLOT), i32)
    mt = mt.at[:, 0].set(C.M_SELL)
    mn = mn.at[:, 0].set(100)
    buy = (seeds < 20) & (money >= 300)
    mt = mt.at[:, 1].set(jnp.where(buy, C.M_BUY_SEED, 0))
    mn = mn.at[:, 1].set(jnp.where(buy, 20 - seeds, 0))
    hire = (hour == 0) & (money > 1500)
    for k in (2, 3, 4):
        mt = mt.at[:, k].set(jnp.where(hire, C.M_HIRE, 0))
        mn = mn.at[:, k].set(jnp.where(hire, 1, 0))
    land = (hour == 1) & (money > 3500) & (g.n_quads[:, p] < 4)
    mt = mt.at[:, 5].set(jnp.where(land, C.M_BUY_LAND, 0))
    mn = mn.at[:, 5].set(jnp.where(land, 1, 0))
    return Actions(op, uarg, un, mt, mi, mn)


def run_arm(B, words, init_seed, cfg):
    tb = E.tables()
    stepf = E.make_step(cfg, with_info=True)

    @jax.jit
    def roll(st, words):
        def body(st, _):
            a0 = wheat_bot(st.g, st.t, 0)
            a1 = wheat_bot(st.g, st.t, 1)
            st, info = stepf(tb, st, a0, a1, words)
            return st, info
        return lax.scan(body, st, None, length=C.LAST_STEP + 1)

    st, info = roll(E.init(B, seed=init_seed), words)
    weeds, empty, shop = (np.asarray(x) for x in info)                       # [719,B,2], [719,B,2], [719,B]
    days = np.arange(C.LAST_STEP + 1)
    eod = (days + 1) % 24 == 0
    return {"money": np.asarray(st.g.money), "weeds": weeds[eod], "empty": empty[eod], "shop": shop[eod],
            "shops": np.asarray(st.g.shops), "n_quads": np.asarray(st.g.n_quads), "oob": np.asarray(st.g.oob)}


def compare(a, b):
    out = {}

    def two(name, x, y):
        ks = stats.ks_2samp(x, y)
        tt = stats.ttest_ind(x, y, equal_var=False)
        out[name] = {"mean_exact": float(np.mean(x)), "mean_train": float(np.mean(y)), "sd_exact": float(np.std(x)),
                     "sd_train": float(np.std(y)), "ks_p": float(ks.pvalue), "welch_p": float(tt.pvalue)}

    two("final_money_seat0", a["money"][:, 0], b["money"][:, 0])
    two("final_money_seat1", a["money"][:, 1], b["money"][:, 1])
    two("final_money_total", a["money"].sum(1), b["money"].sum(1))
    two("weeds_per_game", a["weeds"].sum((0, 2)), b["weeds"].sum((0, 2)))
    two("empty_tiles_rolled_per_game", a["empty"].sum((0, 2)), b["empty"].sum((0, 2)))
    two("land_quadrants_final", a["n_quads"].sum(1), b["n_quads"].sum(1))
    # per-day weeds: mean profile + KS per day, and a Fisher combination of the per-day KS p-values
    pd = []
    for d in range(a["weeds"].shape[0]):
        x, y = a["weeds"][d].sum(1), b["weeds"][d].sum(1)
        pd.append(float(stats.mannwhitneyu(x, y).pvalue) if (x.any() or y.any()) else 1.0)
    out["weeds_per_day"] = {"mean_exact": a["weeds"].sum(2).mean(1).round(3).tolist(),
                            "mean_train": b["weeds"].sum(2).mean(1).round(3).tolist(),
                            "per_day_mannwhitney_p_min": float(np.min(pd)),
                            "fisher_combined_p": float(stats.combine_pvalues(pd).pvalue)}
    # pooled weed rate per rolled empty tile
    for nm, arm in (("exact", a), ("train", b)):
        k, n = int(arm["weeds"].sum()), int(arm["empty"].sum())
        out[f"weed_rate_{nm}"] = {"weeds": k, "rolls": n, "rate": k / n,
                                  "binom_p_vs_0.005": float(stats.binomtest(k, n, 0.005).pvalue)}
    ka, na = out["weed_rate_exact"]["weeds"], out["weed_rate_exact"]["rolls"]
    kb, nb = out["weed_rate_train"]["weeds"], out["weed_rate_train"]["rolls"]
    out["weed_rate_two_sample_p"] = float(stats.chi2_contingency([[ka, na - ka], [kb, nb - kb]])[1])
    # shops per unlock index
    sp = []
    for k in range(C.MAX_SHOPS):
        ca = np.bincount(a["shops"][:, k], minlength=8)
        cb = np.bincount(b["shops"][:, k], minlength=8)
        p_h = float(stats.chi2_contingency([ca, cb])[1])
        p_ua = float(stats.chisquare(ca).pvalue)
        p_ub = float(stats.chisquare(cb).pvalue)
        sp.append({"unlock": k, "exact": ca.tolist(), "train": cb.tolist(), "homog_p": p_h, "uniform_p_exact": p_ua,
                   "uniform_p_train": p_ub})
    out["shops"] = sp
    ja = np.bincount(a["shops"][:, 0] * 8 + a["shops"][:, 1], minlength=64)
    jb = np.bincount(b["shops"][:, 0] * 8 + b["shops"][:, 1], minlength=64)
    keep = (ja + jb) > 0
    out["shop_pair_01_homog_p"] = float(stats.chi2_contingency([ja[keep], jb[keep]])[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--cfg", default="gpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = {"seq": E.CFG_SEQ, "gpu": E.CFG_GPU}[args.cfg]
    B = args.n
    print("devices", jax.devices(), "B", B, flush=True)
    seeds = np.random.RandomState(20260927).randint(0, 2 ** 31, size=B)
    t0 = time.time()
    W = jnp.asarray(np.moveaxis(R.batch_words(seeds), 0, 1))                   # [B,30,624]
    print(f"exact words for {B} seeds: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    a = run_arm(B, W, 0, cfg)
    print(f"exact arm: {time.time() - t0:.0f}s", flush=True)
    del W
    t0 = time.time()
    b = run_arm(B, None, 777, cfg)
    print(f"train arm: {time.time() - t0:.0f}s", flush=True)
    res = compare(a, b)
    res["n_per_arm"] = B
    res["oob_games"] = [int(a["oob"].sum()), int(b["oob"].sum())]
    res["training_mode_randomness"] = ("engine.init(B, seed): every game starts from the identical official initial "
                                       "state (the game has no per-seed initial variation); per-game keys = "
                                       "jax.random.split(PRNGKey(seed), B). step(words=None) consumes the key ONLY at "
                                       "day ends: key, sub = split(key); words = jax.random.bits(sub, (624,), uint32) "
                                       "(uniform). No other randomness exists in reset or step.")
    print(json.dumps(res, indent=1)[:6000])
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
