"""A tiny stand-in for the JAX engine + encoder with the batched Backend interface (backend.py), so the trainer can be
exercised end to end without the real engine.  NOT the game: a toy with a learnable objective.

Toy rules (per seat): up to 16 units on a 10x10 board, every tile has a code 0..3 whose "right" op is WATER / HARVEST /
DIG / CARE; a unit doing the right op on its tile earns $10 and the tile gets a new random code.  Moves walk the board.
Market: SELL of a product that is "hot" today earns $15, otherwise costs $5; HIRE adds a unit for $20.  The money
margin is the reward, as in the real trainer.  The encoder writes tile codes, units and hot flags into the real input
shapes (planes uint8[47,10,10], glob[179], uin[16,76] with the previous-op one-hot at column 32), so the real network
(kagjax/policy/net.py) can learn it.  Lockstep like the real engine: one shared step t; the episode has
`episode_slots` slots (default 48: steps 0..46 played, 47 = dummy).
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import actions as A
from backend import Obs, Common, NU, N_OPS, C_PLANES, G6, U_FEAT, PS

GOOD = jnp.asarray([A.OP_WATER, A.OP_HARVEST, A.OP_DIG, A.OP_CARE], jnp.int32)
MOVE_D = jnp.asarray([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], jnp.int32)       # PASS N S E W


class MGame(NamedTuple):      # one game (vmapped: leading [B])
    money: jnp.ndarray      # float32[2]
    pos: jnp.ndarray        # int32[2,16,2]
    n_units: jnp.ndarray    # int32[2]
    inv: jnp.ndarray        # int32[2,16,12]
    shed: jnp.ndarray       # int32[2,12]
    tile: jnp.ndarray       # int32[2,100]
    hot: jnp.ndarray        # bool[9]
    prices: jnp.ndarray     # int32[9]
    key: jnp.ndarray


class MState(NamedTuple):
    t: jnp.ndarray          # int32[] shared step
    g: MGame                # [B, ...]


class MRaw(NamedTuple):     # one seat's view (farm axis 0 = own)
    t: jnp.ndarray
    g: MGame


class MockBackend(Common):
    def __init__(self, episode_slots=48, **kw):
        self.episode_slots = int(episode_slots)
        self.last_step = self.episode_slots - 2

    # ---- per game
    def _reset1(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return MGame(money=jnp.full(2, 3000.0, jnp.float32), pos=jnp.full((2, NU, 2), 4, jnp.int32), n_units=jnp.full(2, 2, jnp.int32),
                     inv=jnp.zeros((2, NU, 12), jnp.int32), shed=jnp.zeros((2, 12), jnp.int32).at[:, 0].set(5),
                     tile=jax.random.randint(k1, (2, 100), 0, 4), hot=jax.random.bernoulli(k2, 0.4, (9,)),
                     prices=jnp.asarray([25, 35, 60, 120, 250, 50, 160, 200, 100], jnp.int32), key=k3)

    def _seat(self, g, p, a, key):
        k = jnp.arange(NU)
        live = k < g.n_units[p]
        op = jnp.where(live, a.u_op, A.OP_PASS)
        mv = jnp.where(op <= A.OP_WEST, op, 0)
        pos = jnp.clip(g.pos[p] + MOVE_D[mv], 0, 9)
        cell = g.pos[p][:, 1] * 10 + g.pos[p][:, 0]
        code = g.tile[p][cell]
        good = live & (op == GOOD[code])
        gain = 10.0 * good.sum()
        newc = jax.random.randint(key, (NU,), 0, 4)
        tile = g.tile[p].at[jnp.where(good, cell, 100)].set(newc, mode="drop")
        j = jnp.arange(A.NO)
        valid = j < a.n_ord
        sell = valid & (a.o_op == A.M_SELL)
        gain += jnp.where(sell, jnp.where(g.hot[jnp.clip(a.o_item, 0, 8)], 15.0, -5.0), 0.0).sum()
        hires = (valid & (a.o_op == A.M_HIRE)).sum()
        n_new = jnp.minimum(NU, g.n_units[p] + hires)
        gain -= 20.0 * (n_new - g.n_units[p])
        pos = jnp.where((k >= g.n_units[p])[:, None], 4, pos)
        return pos, n_new, tile, gain

    def _step1(self, g, a0, a1, t):
        key, k0, k1, k2 = jax.random.split(g.key, 4)
        p0, n0, t0, g0 = self._seat(g, 0, a0, k0)
        p1, n1, t1, g1 = self._seat(g, 1, a1, k1)
        hot = jnp.where((t + 1) % 24 == 0, jax.random.bernoulli(k2, 0.4, (9,)), g.hot)
        return g._replace(money=g.money + jnp.stack([g0, g1]), pos=jnp.stack([p0, p1]), n_units=jnp.stack([n0, n1]),
                          tile=jnp.stack([t0, t1]), hot=hot, key=key)

    # ---- batched interface
    def reset(self, key, n):
        return MState(t=jnp.int32(0), g=jax.vmap(self._reset1)(jax.random.split(key, n)))

    def consts(self):
        return {}

    def step(self, s, act0, act1, cs=None):
        live = s.t <= self.last_step
        g2 = jax.vmap(self._step1, in_axes=(0, 0, 0, None))(s.g, act0, act1, s.t)
        g2 = jax.tree_util.tree_map(lambda n_, o_: jnp.where(live, n_, o_), g2, s.g)
        return MState(t=jnp.where(live, s.t + 1, s.t), g=g2)

    def t(self, s):
        return s.t

    def money(self, s):
        return s.g.money

    def observe(self, s, seat, cs=None):
        idx = jnp.asarray([seat, 1 - seat])
        g = s.g
        g = g._replace(money=g.money[:, idx], pos=g.pos[:, idx], n_units=g.n_units[:, idx], inv=g.inv[:, idx],
                       shed=g.shed[:, idx], tile=g.tile[:, idx])
        return MRaw(t=jnp.broadcast_to(s.t, g.money.shape[:1]), g=g)

    def state_spec(self, axis):
        return MState(t=PS(), g=jax.tree_util.tree_map(lambda _: PS(axis), self._reset1(jax.random.PRNGKey(0))))

    def encode(self, raw, prev_ops, cs=None):
        return jax.vmap(self._encode1)(raw, prev_ops)

    def _encode1(self, raw, prev_ops):
        g = raw.g
        planes = jnp.zeros((C_PLANES, 10, 10), jnp.uint8)
        codes = g.tile[0].reshape(10, 10)
        for c in range(4):
            planes = planes.at[c].set(jnp.where(codes == c, 255, 0).astype(jnp.uint8))
        nu = jnp.minimum(NU, g.n_units[0])
        k = jnp.arange(NU)
        live = k < nu
        pos = jnp.where(live[:, None], g.pos[0], 0)
        occ = jnp.zeros(100, jnp.int32).at[jnp.where(live, pos[:, 1] * 10 + pos[:, 0], 100)].add(85, mode="drop")
        planes = planes.at[28].set(jnp.minimum(255, occ).reshape(10, 10).astype(jnp.uint8))
        glob = jnp.zeros(G6, jnp.float32)
        glob = glob.at[0].set(raw.t / 720.0).at[2:11].set(g.hot.astype(jnp.float32))
        glob = glob.at[11].set(jnp.clip((g.money[0] - g.money[1]) / 2e4, -3, 3)).at[12].set(nu / 16.0)
        uin = jnp.zeros((NU, U_FEAT), jnp.float32)
        uin = uin.at[k, k].set(1.0).at[:, 16].set(pos[:, 0] / 9.0).at[:, 17].set(pos[:, 1] / 9.0)
        pv = jnp.where(live & (prev_ops >= 0), 32 + prev_ops, U_FEAT)
        uin = uin.at[k, pv].set(1.0, mode="drop")
        uin = jnp.where(live[:, None], uin, 0.0)
        return Obs(planes, glob, pos.astype(jnp.int32), uin, nu.astype(jnp.int32))

    def legal(self, raw):
        def one(g):
            nu = jnp.minimum(NU, g.n_units[0])
            pos = g.pos[0]
            m = jnp.zeros((NU, N_OPS), bool).at[:, 0].set(True)
            m = m.at[:, 1].set(pos[:, 1] > 0).at[:, 2].set(pos[:, 1] < 9).at[:, 3].set(pos[:, 0] < 9).at[:, 4].set(pos[:, 0] > 0)
            for c in (5, 6, 8, 13):                                           # WATER HARVEST DIG CARE (policy classes)
                m = m.at[:, c].set(True)
            return m & (jnp.arange(NU) < nu)[:, None]
        return jax.vmap(one)(raw.g)

    def view(self, raw):
        def one(t, g):
            live = jnp.arange(NU) < g.n_units[0]
            return A.View(step=t, money=g.money[0], nq=jnp.int32(2), n_units=g.n_units[0], pos=g.pos[0], inv=g.inv[0],
                          carried=jnp.where(live[:, None], g.inv[0], 0).sum().astype(jnp.int32), shed=g.shed[0],
                          seeds=jnp.zeros(5, jnp.int32), t_yield=jnp.zeros(NU, jnp.int32), t_dict=jnp.zeros(NU, bool),
                          t_animal=jnp.zeros(NU, bool), t_fert=jnp.zeros(NU, bool), prices=g.prices, sheep_total=jnp.int32(0))
        return jax.vmap(one)(raw.t, raw.g)
