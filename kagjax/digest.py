"""Canonical per-game state digest, identical for the JAX engine (any device, any batch) and for the C++ reference
(via its observe() dicts). Used to verify GPU = CPU = C++ at every step for thousands of games cheaply.

The canonical vector of a game is the complete game state as the C++ exposes it:
  money[2], tiles[2,100,13] (all fields; hidden ones are canonical in both engines), ux/uy of live units (0 for
  unit slots >= n_units), n_units[2], n_quads[2], hires[2], shed[2,12], seeds[2,5], inv[2,40,12],
  insertion RANK of every held item (0 when not held), minv[9], shops[8], n_shops.
Digest = two independent random linear hashes mod 2^32 (uint32[2]); a mismatch in any single field changes both.
"""
import numpy as np
import jax
import jax.numpy as jnp

try:
    from . import consts as C
    from . import state as S
except ImportError:
    import consts as C
    import state as S

_L = 2 + 2 * C.NT * C.NF + 2 * C.U * 2 + 6 + 2 * C.N_ITEMS + 2 * C.N_CROPS + 2 * (2 * C.U * C.N_ITEMS) + 9 + 8 + 1
_rs = np.random.RandomState(20260926)
W = (_rs.randint(0, 2 ** 31, size=(2, _L)).astype(np.uint64) * 2 + 1).astype(np.uint32)   # odd weights


def _canon_np(xp, g):
    """Canonical int32 matrix [B, L] from a batched GameState (xp = jnp or np)."""
    B = g.money.shape[0]
    live = xp.arange(C.U)[None, None, :] < g.n_units[:, :, None]
    ux = xp.where(live, g.ux, 0)
    uy = xp.where(live, g.uy, 0)
    held = g.inv > 0
    rank = xp.sum((held[..., None, :] & (g.ins[..., None, :] < g.ins[..., :, None])).astype(np.int32), axis=-1)
    rank = xp.where(held, rank, 0)
    parts = [g.money, g.tiles, ux, uy, g.n_units, g.n_quads, g.hires, g.shed, g.seeds, g.inv, rank, g.minv, g.shops,
             g.n_shops.reshape(B, 1)]
    return xp.concatenate([xp.asarray(p).astype(np.int32).reshape(B, -1) for p in parts], axis=1)


@jax.jit
def digest(g):
    """uint32[B,2] digest of a batched GameState (JAX, any device)."""
    v = _canon_np(jnp, g).astype(jnp.uint32)
    w = jnp.asarray(W)
    return jnp.stack([jnp.sum(v * w[0][None, :], axis=1, dtype=jnp.uint32),
                      jnp.sum(v * w[1][None, :], axis=1, dtype=jnp.uint32)], axis=1)


def digest_np(g):
    """Same digest in numpy (for states built from C++ observations)."""
    v = _canon_np(np, g).astype(np.uint32)
    with np.errstate(over="ignore"):
        h0 = np.sum(v * W[0][None, :], axis=1, dtype=np.uint32)
        h1 = np.sum(v * W[1][None, :], axis=1, dtype=np.uint32)
    return np.stack([h0, h1], axis=1)


def ref_digest(games):
    """games: list of kagsim_trace.Game -> uint32[len,2] digests of their current states (through observe())."""
    gs = [S.obs_to_game(g.observe(0), g.observe(1))[0] for g in games]
    return digest_np(S.stack_games(gs))


assert _L == _canon_np(np, S.stack_games([S.initial_game()])).shape[1]
