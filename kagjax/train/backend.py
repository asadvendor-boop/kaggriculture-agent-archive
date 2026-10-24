"""The interface the self-play trainer needs from the JAX engine (Agent A: kagjax/engine.py, state.py) and the JAX policy
port (Agent C: kagjax/policy/), and the loader that picks the real implementation or the mock.

The engine is batch-lockstep: every game of a batch is at the same step t (A's EnvState.t is one scalar), so all
functions here are BATCHED (leading dim B = games):

  Backend.episode_slots   int   720 slots per episode: steps t = 0..718 are played, t = 719 is a dummy slot (the
                                engine is done after step 718; stepping a done state is a no-op)
  Backend.last_step       int   718
  Backend.reset(key, n)                    -> state        n fresh games (training randomness from key)
  Backend.consts()                         -> pytree       device tables passed into every jitted program as an
                                                           argument (A's price tables, C's encoder tables)
  Backend.step(state, act0, act1, cs)      -> state        act0 / act1: actions.Act with leading [B] (seat 0 / 1)
  Backend.t(state)                         -> int32[]      the shared step index 0..719
  Backend.money(state)                     -> float32[B,2] banks (== the final reward once done)
  Backend.observe(state, seat, cs)         -> raw          per-seat observation arrays (seat = python int 0 / 1);
                                                           a pytree with leading [B] (C's RawObs for the real env)
  Backend.encode(raw, prev_ops, cs)        -> Obs          network inputs (features6 layout), prev_ops int32[B,16]
  Backend.legal(raw)                       -> bool[B,16,44]  agent.legal_mask per unit
  Backend.view(raw)                        -> actions.View [B]  what the rule layer / qty clamp read
  Backend.state_spec(axis)                 -> pytree of PartitionSpec for the state (multi-device: games sharded)
  Backend.net_apply(params, planes, glob, pos, uin, team, precision) -> dict(op, qty, mk, v)
  Backend.load_params(path, which)         -> (params, meta)
Obs: planes uint8[B,47,10,10], glob float32[B,179], pos int32[B,16,2] (x, y), uin float32[B,16,76], nu int32[B]
(units the network sees = min(16, n_units); rows >= nu are zero as live_rollout pads them).
"""
import os, sys
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
KAGJAX = os.path.dirname(HERE)
ROOT = os.path.dirname(KAGJAX)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import actions as A

NU, N_OPS, N_QTY, N_HEADS, N_MK = 16, 44, 12, 21, 249
C_PLANES, G6, U_FEAT = 47, 179, 76
PS = jax.sharding.PartitionSpec


class Obs(NamedTuple):
    planes: jnp.ndarray     # uint8 [B,47,10,10]
    glob: jnp.ndarray       # float32 [B,G]
    pos: jnp.ndarray        # int32 [B,16,2]
    uin: jnp.ndarray        # float32 [B,16,76]
    nu: jnp.ndarray         # int32 [B]


def init_params(key, width=16, blocks=1, n_teams=48, g_in=G6, n_aux=0, gdim=160, hid=256, team_dim=16):
    """Random params in kagjax.policy.net layout (torch default-like uniform init).  For the mock / small value nets."""
    ks = iter(jax.random.split(key, 64))

    def lin(i, o):
        b = 1.0 / np.sqrt(i)
        return {"w": jax.random.uniform(next(ks), (i, o), jnp.float32, -b, b), "b": jax.random.uniform(next(ks), (o,), jnp.float32, -b, b)}

    def conv(i, o, n=None):
        b = 1.0 / np.sqrt(i * 9)
        sh = (3, 3, i, o) if n is None else (n, 3, 3, i, o)
        bs = (o,) if n is None else (n, o)
        return jax.random.uniform(next(ks), sh, jnp.float32, -b, b), jax.random.uniform(next(ks), bs, jnp.float32, -b, b)

    yy, xx = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    p = {"coords": jnp.asarray(np.stack([xx / 9.0, yy / 9.0], -1).astype(np.float32))}
    if n_teams:
        p["team"] = jax.random.normal(next(ks), (n_teams, team_dim), jnp.float32)
    w, b = conv(C_PLANES + 2, width); p["stem"] = {"w": w, "b": b}
    c1w, c1b = conv(width, width, blocks); c2w, c2b = conv(width, width, blocks)
    p["blocks"] = {"c1w": c1w, "c1b": c1b, "c2w": c2w, "c2b": c2b}
    p["g1"] = lin(g_in + (team_dim if n_teams else 0), gdim); p["g2"] = lin(gdim, gdim); p["film"] = lin(gdim, width)
    p["u1"] = lin(width + gdim + U_FEAT, hid); p["u2"] = lin(hid, hid); p["uop"] = lin(hid, N_OPS); p["uqty"] = lin(hid, N_QTY)
    p["m1"] = lin(2 * width + gdim, hid); p["m2"] = lin(hid, hid); p["mk"] = lin(hid, N_MK); p["v"] = lin(hid, 1)
    if n_aux:
        p["aux"] = lin(hid, n_aux)
    return p


def _net_module():
    from kagjax.policy import net as N         # Agent C's port
    return N


class Common:
    episode_slots = 720
    last_step = 718

    def net_apply(self, params, planes, glob, pos, uin, team, precision=None, with_hm=False):
        """team: int [N] (per-row style ids) or a scalar.  with_hm: also return hm, the market-trunk features the value
        head reads (out["hm"] [N,256])."""
        N = _net_module()
        if precision == "bf16":                  # bf16 weights / activations (rollouts only), fp32 logits
            cast = lambda t: t.astype(jnp.bfloat16) if t.dtype == jnp.float32 else t
            out = N.forward(jax.tree_util.tree_map(cast, params), planes.astype(jnp.bfloat16), glob.astype(jnp.bfloat16), pos,
                            uin.astype(jnp.bfloat16), team, precision=None, with_hm=with_hm)
            return jax.tree_util.tree_map(lambda t: t.astype(jnp.float32), out)
        prec = jax.lax.Precision.HIGHEST if precision == "highest" else None
        return N.forward(params, planes, glob, pos, uin, team, precision=prec, with_hm=with_hm)

    def load_params(self, path, which="state"):
        if path.endswith(".npz"):
            import ckpt_io
            return ckpt_io.load_params_npz(path, which)
        return _net_module().load_ckpt(path, which)

    def init_params(self, key, **kw):
        return init_params(key, **kw)


def load_backend(kind=None, **kw):
    """kind: "mock" | "real" (default: env KAGJAX_BACKEND, else "real" when kagjax/engine.py exists)."""
    kind = kind or os.environ.get("KAGJAX_BACKEND") or ("real" if os.path.exists(os.path.join(KAGJAX, "engine.py")) else "mock")
    if kind == "mock":
        import mock_env
        return mock_env.MockBackend(**kw)
    import real_backend
    return real_backend.RealBackend(**kw)
