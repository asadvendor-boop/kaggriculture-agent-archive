"""Independent numeric check of the JAX trainer's GAE / returns / advantage normalization against the reference
per-trajectory loop of il/ppo_live.py (lines 244-252), with the JAX layout: 720 slots per episode, slot 719 = the
dummy (invalid) slot, done at slot 718; plus truncated segments (SEG=360) with bootstrap.
run (Mac, 1 core): JAX_CACHE_DIR=off kagjax/env1.sh kagjax/verify/check_gae.py
"""
import os
import sys
os.environ.setdefault("JAX_CACHE_DIR", "off")          # never write a compilation cache into kagjax/train
sys.path.insert(0, "/Users/asad.ali/kaggriculture/kagjax/train")
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
import ppo as P                                          # noqa: E402

rng = np.random.RandomState(0)
L, B, LAM = 720, 6, 0.95
rew = rng.randn(L, B).astype(np.float32) * 0.01
v = rng.randn(L, B).astype(np.float32) * 0.1
t = np.arange(L)
valid = np.broadcast_to((t <= 718)[:, None], (L, B)).copy()
done = np.broadcast_to((t == 718)[:, None], (L, B)).copy()
rew[~valid] = 0.0

adv_j = np.asarray(P.gae(jnp.asarray(rew), jnp.asarray(v), jnp.zeros(B), jnp.asarray(done), jnp.asarray(valid), LAM))

# reference: il/ppo_live.py per trajectory (719 real decisions, next value 0 after the last)
adv_r = np.zeros((L, B), np.float32)
for b in range(B):
    r_, v_ = rew[:719, b], v[:719, b]
    last = 0.0
    for k in range(718, -1, -1):
        nv = v_[k + 1] if k + 1 < 719 else 0.0
        last = r_[k] + nv - v_[k] + LAM * last
        adv_r[k, b] = last
err = np.abs(adv_j[:719] - adv_r[:719]).max()
print("full episode: max |adv_jax - adv_ppo_live| =", err, " dummy-slot adv (must be 0):", np.abs(adv_j[719]).max())

# truncated segments: SEG = 360 with bootstrap V(s_360), then the second half with the dummy slot
h = 360
a1 = np.asarray(P.gae(jnp.asarray(rew[:h]), jnp.asarray(v[:h]), jnp.asarray(v[h]), jnp.asarray(done[:h]), jnp.asarray(valid[:h]), LAM))
a2 = np.asarray(P.gae(jnp.asarray(rew[h:]), jnp.asarray(v[h:]), jnp.zeros(B), jnp.asarray(done[h:]), jnp.asarray(valid[h:]), LAM))
ref1 = np.zeros((h, B), np.float32)
for b in range(B):
    last = 0.0
    for k in range(h - 1, -1, -1):
        nv = v[k + 1, b]
        last = rew[k, b] + nv - v[k, b] + LAM * (last if k + 1 < h else 0.0)
        ref1[k, b] = last
print("segment 1 (bootstrap): max err", np.abs(a1 - ref1).max(), "| segment 2 == full-episode tail:",
      np.abs(a2[:359] - adv_j[h:719]).max(), "| segment 2 dummy slot:", np.abs(a2[359]).max())

# advantage normalization: JAX uses the unbiased std over valid samples; ppo_live uses numpy's population std
x = adv_j[valid]
n = x.size
print("N =", n, " unbiased/population std ratio =", float(np.std(x, ddof=1) / np.std(x)), " (sqrt(N/(N-1)) =",
      float(np.sqrt(n / (n - 1))), ")  at N = 737,280:", float(np.sqrt(737280 / 737279)))
