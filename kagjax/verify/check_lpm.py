"""Market-fire log-probs of the trainer (kagjax/train/sample.py, the CURRENT file) vs a float64 reference of the
il/ppo.py definition (P(order) and P(fire) clamped to [1e-6, 1-1e-6]), incl. saturated heads; and the check that
the behavior log-prob stored at rollout and the update's log-prob come from the same function.
run (Mac, 1 core): JAX_CACHE_DIR=off kagjax/env1.sh kagjax/verify/check_lpm.py
"""
import os
import sys
os.environ.setdefault("JAX_CACHE_DIR", "off")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/kagjax/train")
import numpy as np
import jax
import jax.numpy as jnp
import sample as SM
import actions as A

rng = np.random.RandomState(0)
N = 4000
mk = rng.randn(N, 249).astype(np.float32) * 3
for h in range(A.N_HEADS):                                  # saturate some heads both ways
    o = A.HEAD_OFF[h]
    sat = rng.rand(N) < 0.3
    mk[sat, o] += rng.choice([-25, -12, 12, 25], sat.sum())
KS, TAU = 10.0, 0.5
fire = rng.rand(N, A.N_HEADS) < 0.5
out = {"op": jnp.zeros((N, 16, 44)), "qty": jnp.zeros((N, 16, 12)), "mk": jnp.asarray(mk)}
lt = SM.logp_terms(out, jnp.ones((N, 16, 44), bool), -jnp.ones((N, 16), jnp.int8), -jnp.ones((N, 16), jnp.int8),
                   jnp.asarray(fire), 0.2, KS, jnp.full(A.N_HEADS, TAU, jnp.float32))
lpm = np.asarray(lt["lpm"], np.float64)
ref = np.zeros((N, A.N_HEADS))
for h, n in enumerate(A.HEAD_SIZES):
    z = mk[:, A.HEAD_OFF[h]:A.HEAD_OFF[h] + n].astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    # 1 - p0 without cancellation in float64: sum of the non-zero classes
    pnz = np.clip(p[:, 1:].sum(1), 1e-6, 1 - 1e-6)
    pf = 1 / (1 + np.exp(-KS * (np.log(pnz / (1 - pnz)) - np.log(TAU / (1 - TAU)))))
    pf = np.clip(pf, 1e-6, 1 - 1e-6)
    ref[:, h] = np.where(fire[:, h], np.log(pf), np.log(1 - pf))
d = np.abs(lpm - ref)
print("MARKET_FORMULA =", SM.MARKET_FORMULA)
print("lpm (trainer, float32) vs float64 reference: max abs", d.max(), " p99.9", np.quantile(d, 0.999))
# the same inputs through decide's sampling probability and logp_terms: identical pf
pf_dec = np.asarray(SM.fire_terms(jnp.asarray(mk), KS, jnp.full(A.N_HEADS, TAU, jnp.float32))[0])
print("decide's P(fire) == logp_terms' P(fire):", bool(np.array_equal(pf_dec, np.asarray(lt["pf"]))))
