"""Decisions and PPO log-probabilities of the policy, batched over games (replicates il/live_rollout._run_batch sampling
and il/ppo.logp_terms exactly, as array code).

Unit ops   : masked logits (illegal -> -1e9); sampled from softmax(z / T), greedy = argmax.
Quantity   : PICKUP_* / PLACE_* units draw a class from softmax(qty / T) (greedy: argmax); recorded unclamped (qact),
             the clamp to the available stock happens in actions.unit_actions.
Market     : head h fires with p = sharpen(P(class > 0)) = sigmoid(KS (logit(clip(1 - p0)) - logit(tau_h))) (greedy:
             1 - p0 > tau_h); the class is 1 + argmax(p[1:]) (deterministic given the fire, so it carries no log-prob).
logp_terms : per-decision log-probs of the recorded (uact, qact, mfire) under a network output (temperature T), plus the
             full unit log-softmax and the fire probabilities for the KL to the reference policy.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import actions as A

NEG = -1e9
_idx = np.full((A.N_HEADS, 18), A.HEAD_OFF[-1], np.int32)
_pad = np.ones((A.N_HEADS, 18), bool)
for _h, _n in enumerate(A.HEAD_SIZES):
    _idx[_h, :_n] = A.HEAD_OFF[_h] + np.arange(_n); _pad[_h, :_n] = False
HEAD_IDX, HEAD_PAD = jnp.asarray(np.minimum(_idx, A.HEAD_OFF[-1] - 1)), jnp.asarray(_pad)


class Dec(NamedTuple):
    op: jnp.ndarray       # int32 [N,16] op class of every unit slot (argmax / sample; meaningless beyond nu)
    uact: jnp.ndarray     # int8  [N,16] recorded op class, -1 beyond nu
    qact: jnp.ndarray     # int8  [N,16] recorded quantity class for PICKUP / PLACE units, else -1
    qty: jnp.ndarray      # int32 [N,16] quantity class used to build the command
    fire: jnp.ndarray     # bool  [N,21]
    classes: jnp.ndarray  # int32 [N,21]


def heads(mk):
    """[N,249] market logits -> [N,21,18] per-head logits, padded classes = -inf."""
    return jnp.where(HEAD_PAD, -jnp.inf, mk[..., HEAD_IDX])


def logit(p):
    return jnp.log(p / (1.0 - p))


MARKET_FORMULA = "stable"      # "stable": logit P(order) = logsumexp(z[1:]) - z[0] (no cancellation when p0 -> 1);
                               # "torch":  logit(clip(1 - softmax(z)[0])) exactly as il/ppo.py (tests compare with torch)
LOGIT_CLAMP = float(np.log((1 - 1e-6) / 1e-6))   # the torch code clamps P(order) and P(fire) to [1e-6, 1 - 1e-6]


def market_logit(mk):
    """[N,249] market logits -> logit P(order) per head [N,21], clamped to +-LOGIT_CLAMP."""
    h = heads(mk)
    if MARKET_FORMULA == "torch":
        pnz = jnp.clip(1.0 - jax.nn.softmax(h, -1)[..., 0], 1e-6, 1 - 1e-6)
        return logit(pnz)
    return jnp.clip(jax.nn.logsumexp(h[..., 1:], -1) - h[..., 0], -LOGIT_CLAMP, LOGIT_CLAMP)


def fire_terms(mk, KS, tau):
    """P(fire) = sharpen(P(order)) per head and its logs: (pf, log pf, log(1 - pf)), all [N,21], pf in [1e-6, 1-1e-6].
    "torch" reproduces il/ppo.logp_terms arithmetic exactly; "stable" computes the same quantity without cancellation."""
    lt = logit(jnp.asarray(tau, jnp.float32))
    if MARKET_FORMULA == "torch":
        pf = jnp.clip(jax.nn.sigmoid(KS * (market_logit(mk) - lt)), 1e-6, 1 - 1e-6)
        return pf, jnp.log(pf), jnp.log(1 - pf)
    a = jnp.clip(KS * (market_logit(mk) - lt), -LOGIT_CLAMP, LOGIT_CLAMP)
    return jax.nn.sigmoid(a), jax.nn.log_sigmoid(a), jax.nn.log_sigmoid(-a)


def decide(out, umask, nu, key, T, KS, tau, greedy):
    """out: dict(op [N,16,44], qty [N,16,12], mk [N,249]); umask bool [N,16,44]; nu int [N]; tau [21] or [N,21] market
    thresholds; greedy: python bool (static) or bool [N] (per game).  Returns Dec.  The sampling distributions are
    exactly those whose log-probabilities logp_terms returns (same formulas, same clamps)."""
    k1, k2, k3 = jax.random.split(key, 3)
    z = jnp.where(umask, out["op"], NEG)
    live = jnp.arange(z.shape[1])[None, :] < nu[:, None]
    g_op = jnp.argmax(z, -1)
    g_q = jnp.argmax(out["qty"], -1)
    h = heads(out["mk"])
    cls_nz = 1 + jnp.argmax(h[..., 1:], -1)                   # most likely non-zero class
    if MARKET_FORMULA == "torch":
        g_fire = 1.0 - jax.nn.softmax(h, -1)[..., 0] > tau       # live_rollout / np_runtime greedy test
    else:
        g_fire = jax.nn.logsumexp(h[..., 1:], -1) - h[..., 0] > logit(jnp.asarray(tau, jnp.float32))   # same test, stable
    if greedy is True:
        op, q, fire = g_op, g_q, g_fire
    else:
        s_op = jax.random.categorical(k1, jnp.where(umask, out["op"] / T, NEG), -1)
        s_q = jax.random.categorical(k2, out["qty"] / T, -1)
        s_fire = jax.random.uniform(k3, g_fire.shape) < fire_terms(out["mk"], KS, tau)[0]
        if greedy is False:
            op, q, fire = s_op, s_q, s_fire
        else:
            gr = jnp.asarray(greedy)
            op = jnp.where(gr[:, None], g_op, s_op); q = jnp.where(gr[:, None], g_q, s_q)
            fire = jnp.where(gr[:, None], g_fire, s_fire)
    cls = jnp.where(fire, cls_nz, 0)
    pp = (op >= A.OP_PICKUP0) & live
    return Dec(op=op.astype(jnp.int32), uact=jnp.where(live, op, -1).astype(jnp.int8), qact=jnp.where(pp, q, -1).astype(jnp.int8),
               qty=q.astype(jnp.int32), fire=fire, classes=cls.astype(jnp.int32))


def logp_terms(out, umask, uact, qact, mfire, T, KS, tau):
    """il/ppo.logp_terms: log-probs of the recorded decisions under `out` (temperature T, market threshold tau).
    Used both for the behavior log-probs stored at rollout and for the update (one function, same inputs)."""
    zu = jnp.where(umask, out["op"] / T, NEG)
    lpu_all = jax.nn.log_softmax(zu, -1)
    ua = uact.astype(jnp.int32); qa = qact.astype(jnp.int32)
    valid = ua >= 0
    lpu = jnp.take_along_axis(lpu_all, jnp.maximum(ua, 0)[..., None], -1)[..., 0]
    lpq_all = jax.nn.log_softmax(out["qty"] / T, -1)
    qvalid = qa >= 0
    lpq = jnp.take_along_axis(lpq_all, jnp.maximum(qa, 0)[..., None], -1)[..., 0]
    pf, lp1, lp0 = fire_terms(out["mk"], KS, tau)
    f = mfire.astype(jnp.float32)
    lpm = f * lp1 + (1 - f) * lp0
    return dict(lpu=lpu, valid=valid, lpu_all=lpu_all, lpq=lpq, qvalid=qvalid, lpm=lpm, pf=pf,
                lpq_all=lpq_all, lpf1=lp1, lpf0=lp0)          # (the last three: entropy bonus, ENT_COEF)


def build_act(dec, nu, view, rules):
    """Dec of ONE game (vmap) -> engine-native Act after the rule layer (unit commands, canonical market, rules)."""
    u_op, u_arg, u_n = A.unit_actions(dec.op, dec.qty, nu, view)
    o_op, o_item, o_n, n_ord = A.decode_market(dec.classes)
    a = A.Act(u_op, u_arg, u_n, nu.astype(jnp.int32), o_op, o_item, o_n, n_ord)
    return A.apply_rules(view, a, rules)
