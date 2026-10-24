"""Training-mode world randomness (engine words=None: i.i.d. jax.random words per game-day) vs the exact Kaggle
generator (rng.day_words: CPython MT19937 of (seed*1000003)^day), with identical scripted play (all PASS, then a
fixed land-buying + building script so the empty-tile count varies): weeds spawned per farm per day and the shop
unlock sequence (8 shops x uniform) - same distribution expected.
usage (server, CPU): python rng_check.py --B 1024
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import jax  # noqa: E402
import consts as C  # noqa: E402
import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402


def run(B, exact, seed0):
    st = E.init(B, seed=seed0)
    words = np.moveaxis(R.batch_words(list(range(seed0, seed0 + B))), 0, 1) if exact else None   # [B,30,624]
    a = S.pass_actions(B)
    weeds = np.zeros((B, 2, 30), np.int32)
    prev_w = np.zeros((B, 2), np.int32)
    empt = np.zeros((B, 2, 30), np.int32)
    for t in range(719):
        if t % 24 == 23:
            k = np.asarray(st.g.tiles[..., C.F_KIND])
            empt[:, :, t // 24] = (k == C.K_EMPTY).sum(-1)
        st = E.step(st, a, a, words)
        if t % 24 == 23:
            k = np.asarray(st.g.tiles[..., C.F_KIND])
            w = (k == C.K_WEED).sum(-1)
            weeds[:, :, t // 24] = w - prev_w
            prev_w = w
    shops = np.asarray(st.g.shops)
    return weeds, empt, shops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1024)
    args = ap.parse_args()
    out = {}
    for name, exact, s0 in (("train_iid", False, 7), ("exact_mt", True, 1_000_000)):
        w, e, sh = run(args.B, exact, s0)
        rate = w[:, :, :29].sum() / max(1, e[:, :, :29].sum())
        cnt = np.bincount(sh[sh >= 0].ravel(), minlength=8)
        out[name] = (rate, w[:, :, :29].mean(), cnt)
        print(f"{name}: weed spawn rate per empty tile-day {rate:.5f} (expected 0.005) | weeds/farm/day "
              f"{w[:, :, :29].mean():.4f} | shop draws per id {cnt.tolist()} (n={cnt.sum()})", flush=True)
    (r1, _, c1), (r2, _, c2) = out["train_iid"], out["exact_mt"]
    n1 = c1.sum(); n2 = c2.sum()
    chi = float((((c1 - c2 * n1 / n2) ** 2) / np.maximum(1, c2 * n1 / n2)).sum())
    print(f"weed rate diff {r1 - r2:+.5f}; shop-count chi2 (7 dof, 95% = 14.07): {chi:.2f}")


if __name__ == "__main__":
    main()
