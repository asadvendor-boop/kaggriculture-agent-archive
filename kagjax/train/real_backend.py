"""The real Backend: Agent A's JAX engine (kagjax/engine.py, state.py) + Agent C's JAX encoder / legal masks / network
(kagjax/policy/encode.py, act.py, net.py), glued with array adapters written here:

  raw_from_state(g, t, seat)  A's GameState [B] -> C's RawObs [B] for one seat (own farm first), exactly what
                              policy.encode.raw_from_obs builds from the C++ observation dict (tested by
                              test_real_backend.py on real game states: bit-identical)
  view_from_raw(raw)          C's RawObs (one game) -> actions.View for the rule layer (== actions.obs_to_view(obs))
  to_engine(act)              actions.Act [B] (16 units) -> A's Actions [B] (40 units, 10 slots)
Training randomness: the engine's words=None mode (end-of-day weeds / shop draws from each game's key).
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from backend import Common, Obs, KAGJAX, NU, PS
import actions as A

if KAGJAX not in sys.path:
    sys.path.insert(0, KAGJAX)
import consts as AC                    # Agent A (engine ids)
import engine as AE
import state as AS
from kagjax.policy import encode as CE
from kagjax.policy import act as CA
from kagjax.policy import consts as CK

i32, f32 = jnp.int32, jnp.float32
# A's tile kind (EMPTY LOCKED WEED COOP PASTURE PLANT) -> C's (EMPTY LOCKED WEED PLANT COOP PASTURE)
_KMAP = np.zeros(6, np.int32)
_KMAP[[AC.K_EMPTY, AC.K_LOCKED, AC.K_WEED, AC.K_COOP, AC.K_PASTURE, AC.K_PLANT]] = [CK.K_EMPTY, CK.K_LOCKED, CK.K_WEED, CK.K_COOP, CK.K_PASTURE, CK.K_PLANT]
KMAP = jnp.asarray(_KMAP)
assert CK.ITEMS == AC.ITEM_NAMES and A.ITEMS == AC.ITEM_NAMES


def prices(g, tb):
    """int32[B,9] observed prices = engine.prices, with the tables passed in (never built inside a trace)."""
    return tb.P[jnp.arange(AC.N_PRODUCTS)[None, :] * AC.NTAB + jnp.clip(g.minv - AC.INV_LO, 0, AC.NTAB - 1)]


def raw_from_state(g, t, seat, tb):
    """A's GameState (batched [B,...]) at step t -> C's RawObs [B] for `seat` (python int); tb = engine.tables()."""
    f = jnp.asarray([seat, 1 - seat])
    B = g.money.shape[0]
    tiles = g.tiles[:, f]                                             # [B,2,100,NF]
    col = lambda c: tiles[..., c]
    ka = col(AC.F_KIND)
    plant = ka == AC.K_PLANT
    an = (col(AC.F_ANIMAL) == 1) & ((ka == AC.K_COOP) | (ka == AC.K_PASTURE))
    pa = plant | an
    what = jnp.where(plant, col(AC.F_WHAT), jnp.where(an, col(AC.F_WHAT) - AC.GOOSE, -1))
    sq = lambda x: x.reshape(B, 2, 10, 10)
    shops = jnp.zeros((B, 8), i32).at[jnp.arange(B)[:, None], jnp.where(g.shops >= 0, g.shops, 8)].add(1, mode="drop")
    return CE.RawObs(
        step=jnp.broadcast_to(jnp.asarray(t, i32), (B,)),
        money=g.money[:, f].astype(f32), n_units=g.n_units[:, f],
        unit_xy=jnp.stack([g.ux[:, f], g.uy[:, f]], -1), hires_today=g.hires[:, f], n_quads=g.n_quads[:, f],
        tile_kind=sq(KMAP[ka]), tile_what=sq(what), tile_animal=sq(an),
        tile_day=sq(jnp.where(pa, col(AC.F_DAY), 0)), tile_yield=sq(jnp.where(pa, col(AC.F_YIELD), 0)),
        tile_dry=sq(jnp.where(pa, col(AC.F_DRY), 0)), tile_watered=sq(plant & (col(AC.F_WATERED) == 1)),
        tile_fed=sq(an & (col(AC.F_FED) == 1)), tile_cared=sq(an & (col(AC.F_CARED) == 1)),
        tile_fert=sq(an & (col(AC.F_FERTAV) == 1)), tile_bonus=sq(jnp.where(an, col(AC.F_CARE), 0)),
        tile_fert_until=sq(jnp.where(plant, col(AC.F_FERTU), -1)), tile_life=sq(jnp.where(plant, col(AC.F_MLS), -1)),
        shed=g.shed[:, seat], seeds=g.seeds[:, seat], inv=g.inv[:, seat],
        mkt_inv=g.minv, mkt_price=prices(g, tb), shops=shops)


def view_from_raw(raw):
    """C's RawObs of ONE game -> actions.View."""
    n = raw.n_units[0]
    k = jnp.arange(NU)
    xy = raw.unit_xy[0, :NU]
    x, y = jnp.clip(xy[:, 0], 0, 9), jnp.clip(xy[:, 1], 0, 9)
    kind = raw.tile_kind[0][y, x]
    live = k < n
    t_dict = live & (kind != CK.K_EMPTY) & (kind != CK.K_LOCKED)
    all_live = jnp.arange(raw.inv.shape[0]) < n
    inv_all = jnp.where(all_live[:, None], raw.inv, 0)
    sheep = (raw.tile_animal[0] & (raw.tile_what[0] == CK.ANIMALS.index("SHEEP"))).sum() + raw.shed[A.I_SHEEP] + inv_all[:, A.I_SHEEP].sum()
    return A.View(step=raw.step, money=raw.money[0], nq=raw.n_quads[0], n_units=n, pos=jnp.where((k < n)[:, None], xy, 0),
                  inv=inv_all[:NU], carried=inv_all.sum().astype(i32), shed=raw.shed, seeds=raw.seeds,
                  t_yield=jnp.where(t_dict, raw.tile_yield[0][y, x], 0), t_dict=t_dict, t_animal=live & raw.tile_animal[0][y, x],
                  t_fert=live & raw.tile_fert[0][y, x], prices=raw.mkt_price, sheep_total=sheep.astype(i32))


def to_engine(a):
    """actions.Act [B] -> A's Actions [B] (units padded to 40 with PASS, count 1)."""
    B = a.u_op.shape[0]
    pad = AC.U - A.NU
    p = lambda x, v: jnp.concatenate([x, jnp.full((B, pad), v, i32)], 1)
    return AS.Actions(uop=p(a.u_op, AC.OP_PASS), uarg=p(a.u_arg, 0), un=p(a.u_n, 1), mtype=a.o_op, mitem=a.o_item, mn=a.o_n)


class RealBackend(Common):
    episode_slots = AC.EPISODE_STEPS          # 720
    last_step = AC.LAST_STEP                  # 718

    def __init__(self, **kw):
        self._g0 = AS.initial_game((0, 0))
        # A's engine loop structure (identical results, very different speed): CFG_GPU on accelerators (static trip
        # counts, no conds: 10.6 ms/step at B=4096 vs 110 ms for the sequential default), CFG_SEQ on one CPU core.
        name = os.environ.get("ENGINE_CFG") or ("gpu" if jax.default_backend() != "cpu" else "seq")
        self.loop_cfg = getattr(AE, "CFG_" + name.upper(), None)
        self._impl = getattr(AE, "_step_impl", None)

    def consts(self):
        """Device tables passed INTO the jitted programs (A: exact price tables; C: exact encoder tables)."""
        return {"a": AE.tables(), "c": CE.tables_jnp()}

    def reset(self, key, n):
        keys = jax.random.split(key, n)
        g = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n,) + np.shape(x)), self._g0)
        g = g._replace(key=jax.random.key_data(keys) if jnp.issubdtype(keys.dtype, jax.dtypes.prng_key) else keys.astype(jnp.uint32))
        return AS.EnvState(t=jnp.asarray(0, i32), g=g)

    def step(self, s, act0, act1, cs):
        # cs["words"] (optional, uint32[30,B,624] = rng.batch_words(seeds)): the exact end-of-day randomness of the
        # Kaggle / C++ engine for those seeds (tests); absent = training randomness from each game's key
        words = cs["words"][jnp.clip(s.t // AC.TURNS_PER_DAY, 0, cs["words"].shape[0] - 1)] if "words" in cs else None
        return self.step_engine(s, to_engine(act0), to_engine(act1), words, cs)

    def step_engine(self, s, e0, e1, words, cs):
        """A's step on engine-format Actions (both seats); words uint32[B,624] or None."""
        if self._impl is not None and self.loop_cfg is not None:
            return self._impl(cs["a"], s, e0, e1, words, cfg=self.loop_cfg)
        return AE._step_jit(cs["a"], s, e0, e1, words)                                  # older engine.py

    def step_tapes(self, s, act0, act1, tid, tape_seat, tapes, key, cs):
        """Step where games with tid >= 0 are recorded worlds: seat `tape_seat` replays the tape's actions and the
        end-of-day words are the tape seed's exact words; other games draw random words (same distribution).
        tapes: device dict from tapes.build_bank (optionally with l* arrays = the other seat's recorded actions, for
        the replay test: tapes["replay_both"])."""
        B = s.g.money.shape[0]
        e0, e1 = to_engine(act0), to_engine(act1)
        is_t = tid >= 0
        ti = jnp.maximum(tid, 0)
        t = jnp.minimum(s.t, 719)
        tp = AS.Actions(*(tapes[k][ti, t].astype(i32) for k in ("uop", "uarg", "un", "mtype", "mitem", "mn")))
        sel = lambda m, a, b: jax.tree_util.tree_map(lambda x, y: jnp.where(m[:, None], y, x), a, b)
        e0 = sel(is_t & (tape_seat == 0), e0, tp); e1 = sel(is_t & (tape_seat == 1), e1, tp)
        if "luop" in tapes:                       # test only: the learner seat replays its recorded actions too
            lp = AS.Actions(*(tapes["l" + k][ti, t].astype(i32) for k in ("uop", "uarg", "un", "mtype", "mitem", "mn")))
            e0 = sel(is_t & (tape_seat == 1), e0, lp); e1 = sel(is_t & (tape_seat == 0), e1, lp)
        day = jnp.clip(s.t // AC.TURNS_PER_DAY, 0, 29)
        words = jnp.where(is_t[:, None], tapes["words"][ti, day], jax.random.bits(key, (B, AC.NW_WORDS), jnp.uint32))
        if "shops_rec" not in tapes:                          # unpinned (natural words only)
            return self.step_engine(s, e0, e1, words, cs)
        impl = self._impl

        def pinned(_):
            # C++ reference pinning (sim_trace.hpp end_of_day with forced_shops / weed_prefix): pass 1 gets each farm's
            # empty-tile count E (independent of the words), pass 2 steps again with the day's words rebuilt so that
            #  - leader in seat 1: its weed draws k = E0 .. E0+E1-1 read the recorded positions e0_rec + (k - E0)
            #    (C++: a second generator skipped by weed_prefix[1][day] = e0_rec draws);
            #  - the shop draw (first word at index >= 2(E0+E1) with top 4 bits < 8) is the recorded shop of this
            #    unlock (C++: forced_shops[n_shops]).  Farm 0 and a seat-0 leader use the natural stream, as in C++.
            _, info = impl(cs["a"], s, e0, e1, words, cfg=self.loop_cfg, with_info=True)
            E = info[1]                                                           # [B,2] empty tiles rolled
            E0, E1 = E[:, :1], E[:, 1:]
            w = jnp.arange(AC.NW_WORDS)[None, :]
            k = w // 2
            lead1 = (is_t & (tape_seat == 1))[:, None]
            er = tapes["e0_rec"][ti, day][:, None]
            src = jnp.where(lead1 & (k >= E0) & (k < E0 + E1), 2 * (er + k - E0) + (w % 2), w)
            wp = jnp.take_along_axis(words, jnp.clip(src, 0, AC.NW_WORDS - 1), axis=1)
            shop = tapes["shops_rec"][ti, jnp.clip(s.g.n_shops, 0, 7)][:, None]
            force = is_t[:, None] & (shop >= 0) & (w == 2 * (E0 + E1))
            wp = jnp.where(force, (shop.astype(jnp.uint32) << 28) | (wp & jnp.uint32(0x0FFFFFFF)), wp)
            wp = jnp.where(is_t[:, None], wp, words)
            return impl(cs["a"], s, e0, e1, wp, cfg=self.loop_cfg)

        is_eod = ((s.t + 1) % AC.TURNS_PER_DAY == 0) & (s.t <= AC.LAST_STEP)
        return lax.cond(is_eod, pinned, lambda _: impl(cs["a"], s, e0, e1, words, cfg=self.loop_cfg), None)

    def t(self, s):
        return s.t

    def money(self, s):
        return s.g.money.astype(f32)

    def observe(self, s, seat, cs):
        return raw_from_state(s.g, s.t, seat, cs["a"])

    def encode(self, raw, prev_ops, cs):
        enc = jax.vmap(CE.encode, in_axes=(0, 0, None))(raw, prev_ops, cs["c"])
        return Obs(enc["planes"], enc["glob"], enc["pos"].astype(i32), enc["uf"], enc["valid"].sum(-1).astype(i32))

    def legal(self, raw):
        return jax.vmap(CA.legal_mask)(raw)

    def view(self, raw):
        return jax.vmap(view_from_raw)(raw)

    def state_spec(self, axis):
        g = jax.tree_util.tree_map(lambda _: PS(axis), self._g0)
        return AS.EnvState(t=PS(), g=g)
