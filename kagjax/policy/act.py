"""Batched action selection in JAX: legal masks, greedy / temperature-sampled unit ops, quantity heads with the clamp,
market heads (tau threshold, sharpened Bernoulli when sampling), the canonical market decode, and the log-probs PPO
needs.  Semantics = il/np_runtime.NumpyAgent (greedy) and il/live_rollout._run_batch / il/ppo.py (sampling, PPO terms).
The rule layer (features.midnight_fix: final fix, homing, midnight sale, plan rules) is NOT applied here; the
trainer's kagjax/train/actions.py apply_rules does it on the Act this module returns.

Output action format = kagjax/train/actions.Act (engine-native, mirrors the C++ Action struct):
  u_op int32[16]  engine unit op (0 PASS 1 NORTH 2 SOUTH 3 EAST 4 WEST 5 PICKUP 6 DROP 7 PLACE 8 PLANT 9 WATER 10 HARVEST
                  11 FERTILIZE 12 DIG 13 BUILD_COOP 14 BUILD_PASTURE 15 FEED 16 COLLECT_FERTILIZER 17 CARE)
  u_arg int32[16] crop (PLANT) / item (PICKUP, PLACE), else 0;  u_n int32[16] quantity (PICKUP/PLACE), else 1
  n_cmd int32[]   commands issued = min(16, n_units)
  o_op int32[10]  market op (0 NONE 1 HIRE 2 BUY_LAND 3 BUY_SEED 4 BUY_PRODUCT 5 BUY_ANIMAL 6 SELL), compacted
  o_item int32[10] item (ITEMS index; BUY_SEED: crop index), o_n int32[10] quantity (HIRE / BUY_LAND: 1), n_ord int32[]
legal_mask / decide / decode_market / unit_commands work on ONE game (vmap them); policy_step and logp_terms take
batches.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K
from kagjax.policy import encode as E
from kagjax.policy import net as N

NU = K.MAX_UNITS
NO = 10
NEG = -1e9
LOGIT_CLAMP = float(np.log((1 - 1e-6) / 1e-6))          # ppo.py clamps probabilities to [1e-6, 1 - 1e-6]


class Act(NamedTuple):
    u_op: jnp.ndarray
    u_arg: jnp.ndarray
    u_n: jnp.ndarray
    n_cmd: jnp.ndarray
    o_op: jnp.ndarray
    o_item: jnp.ndarray
    o_n: jnp.ndarray
    n_ord: jnp.ndarray


_MOVE_DXY = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
_OP_UOP = jnp.asarray(K.OP_UOP)
_OP_ARG = jnp.asarray(np.maximum(K.OP_ITEM, 0))
_ANIMAL_HOME = jnp.asarray(K.ANIMAL_HOME_KIND)
_CROP_FIRST = jnp.asarray(K.CROP_FIRST)
_SHED = jnp.asarray(E._SHED_MASK)


# ---------------------------------------------------------------------------------------------- legal mask
def legal_mask(raw):
    """agent.legal_mask / np_runtime.legal_mask_np for the first min(16, n_units) own units -> bool [16, 44].
    Rows of absent units allow PASS only (so masked softmaxes stay finite)."""
    n = jnp.minimum(NU, raw.n_units[0])
    valid = jnp.arange(NU) < n
    x = jnp.clip(raw.unit_xy[0, :NU, 0], 0, 9); y = jnp.clip(raw.unit_xy[0, :NU, 1], 0, 9)
    kind = raw.tile_kind[0][y, x]; what = raw.tile_what[0][y, x]
    animal = raw.tile_animal[0][y, x]
    yu = raw.tile_yield[0][y, x]; tday = raw.tile_day[0][y, x]
    iv = raw.inv[:NU]                                                        # [16,12]
    day = raw.step // 24
    at_shed = _SHED[y, x]
    is_d = kind >= K.K_WEED                                                  # dict tile (not None / LOCKED)
    is_plant = kind == K.K_PLANT
    cols = {}
    cols["PASS"] = jnp.ones(NU, bool)
    for mv, (dx, dy) in _MOVE_DXY.items():
        cols[mv] = (x + dx >= 0) & (x + dx < 10) & (y + dy >= 0) & (y + dy < 10)
    crop_ok = (what >= 0) & (what < 5)
    harvest_plant = is_plant & crop_ok & (yu > 0) & (day - tday >= _CROP_FIRST[jnp.clip(what, 0, 4)])
    animal_t = is_d & animal
    cols["WATER"] = is_plant & ~raw.tile_watered[0][y, x]
    cols["FERTILIZE"] = is_plant & (iv[:, 8] > 0)
    cols["HARVEST"] = harvest_plant | (animal_t & (yu > 0))
    cols["FEED"] = animal_t & ~raw.tile_fed[0][y, x] & (iv[:, 0] > 0)
    cols["COLLECT_FERTILIZER"] = animal_t & raw.tile_fert[0][y, x]
    cols["CARE"] = animal_t & ~raw.tile_cared[0][y, x]
    cols["DIG"] = is_d & ~animal_t
    empty = kind == K.K_EMPTY
    cols["BUILD_COOP"] = empty
    cols["BUILD_PASTURE"] = empty
    for c, crop in enumerate(K.CROPS):
        cols["PLANT_" + crop] = empty & (raw.seeds[c] > 0)
    cols["DROP"] = at_shed & (iv > 0).any(1)
    for i, it in enumerate(K.ITEMS):
        cols["PICKUP_" + it] = at_shed & (raw.shed[i] > 0)
        has = iv[:, i] > 0
        if it in K.ANIMALS:
            a = K.ANIMALS.index(it)
            home = is_d & ~animal_t & (kind == _ANIMAL_HOME[a])
            cols["PLACE_" + it] = has & (home | at_shed)
        else:
            cols["PLACE_" + it] = has & at_shed
    m = jnp.stack([cols[o] for o in K.OPS], 1)                               # [16,44]
    pass_only = jnp.zeros(K.N_OPS, bool).at[0].set(True)
    return jnp.where(valid[:, None], m, pass_only[None, :])


# ---------------------------------------------------------------------------------------------- market heads
_HEAD_IDX = np.zeros((K.N_HEADS, K.MAX_CLASSES), np.int32)
_HEAD_OK = np.zeros((K.N_HEADS, K.MAX_CLASSES), bool)
for _h, _n in enumerate(K.HEAD_SIZES):
    _HEAD_IDX[_h, :_n] = K.HEAD_OFF[_h] + np.arange(_n)
    _HEAD_OK[_h, :_n] = True
_HEAD_IDX_J = jnp.asarray(_HEAD_IDX); _HEAD_OK_J = jnp.asarray(_HEAD_OK)


def head_logits(mk):
    """mk [249] -> padded per-head logits [21, 18] (-inf beyond each head's size)."""
    return jnp.where(_HEAD_OK_J, mk[_HEAD_IDX_J], -jnp.inf)


def market_logit_order(mk):
    """logit(P(order)) = log(1 - p0) - log(p0) per head, computed as logsumexp(z[1:]) - z[0] (no cancellation)."""
    z = head_logits(mk)
    return jax.nn.logsumexp(z[:, 1:], axis=1) - z[:, 0]


def market_class(mk):
    """Most likely non-zero class per head: 1 + argmax(z[1:])."""
    z = head_logits(mk)
    return 1 + jnp.argmax(z[:, 1:], axis=1)


def _tau_vec(tau):
    t = jnp.asarray(tau, jnp.float32)
    return jnp.broadcast_to(t, (K.N_HEADS,))


def _logit(p):
    return jnp.log(p) - jnp.log1p(-p)


def market_fire_logits(mk, tau, ks):
    """Sampling policy of a market head (rollout.sharpen / ppo.logp_terms): P(fire) = sigmoid(ks * (logit(pnz) -
    logit(tau))) with pnz clamped to [1e-6, 1-1e-6] -> the argument a of that sigmoid [21]."""
    d = jnp.clip(market_logit_order(mk), -LOGIT_CLAMP, LOGIT_CLAMP)
    return ks * (d - _logit(_tau_vec(tau)))


# ---------------------------------------------------------------------------------------------- market decode
_SELL_ORDER = ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]
_cand = []                                                  # (head, mop, item, kind): kind 0 plain, 1 hire slot j, 2 land
for _p in _SELL_ORDER:
    _cand.append((K.HEAD_INDEX["SELL:" + _p], K.MOP["SELL"], K.ITEMS.index(_p), 0, 0))
for _j in range(10):
    _cand.append((K.HEAD_INDEX["HIRE"], K.MOP["HIRE"], 0, 1, _j))
_cand.append((K.HEAD_INDEX["BUY_LAND"], K.MOP["BUY_LAND"], 0, 2, 0))
for _a in K.ANIMALS:
    _cand.append((K.HEAD_INDEX["BUY_ANIMAL:" + _a], K.MOP["BUY_ANIMAL"], K.ITEMS.index(_a), 0, 0))
for _c in K.CROPS:
    _cand.append((K.HEAD_INDEX["BUY_SEED:" + _c], K.MOP["BUY_SEED"], K.CROPS.index(_c), 0, 0))
_cand.append((K.HEAD_INDEX["BUY_PRODUCT:WHEAT"], K.MOP["BUY_PRODUCT"], K.ITEMS.index("WHEAT"), 0, 0))
_cand.append((K.HEAD_INDEX["BUY_PRODUCT:FERTILIZER"], K.MOP["BUY_PRODUCT"], K.ITEMS.index("FERTILIZER"), 0, 0))
_C_HEAD = jnp.asarray([c[0] for c in _cand], jnp.int32); _C_OP = jnp.asarray([c[1] for c in _cand], jnp.int32)
_C_ITEM = jnp.asarray([c[2] for c in _cand], jnp.int32); _C_KIND = jnp.asarray([c[3] for c in _cand], jnp.int32)
_C_SLOT = jnp.asarray([c[4] for c in _cand], jnp.int32)
_C_BIG = jnp.asarray([K.BIG["SELL"] if c[1] == K.MOP["SELL"] else
                      (K.BIG["BUY_PRODUCT:WHEAT"] if c[0] == K.HEAD_INDEX["BUY_PRODUCT:WHEAT"] else 0) for c in _cand], jnp.int32)


def decode_market(classes):
    """features.decode_market (SELL_ORDER canonical): classes int [21] -> (o_op, o_item, o_n, n_ord).  Order: sells
    (fixed product order), hires, land, animals, seeds, product buys; more than 10 orders -> trailing SELLs dropped
    first, never a HIRE / purchase (the reference keeps sells[:10 - len(other)] + other[:10])."""
    c = classes[_C_HEAD]
    active = jnp.where(_C_KIND == 1, c > _C_SLOT, c > 0)
    qty = jnp.where(_C_KIND >= 1, 1, jnp.where((c == 17) & (_C_BIG > 0), _C_BIG, c))
    is_sell = _C_OP == K.MOP["SELL"]
    other = active & ~is_sell
    other_rank = jnp.cumsum(other) - other
    keep_other = other & (other_rank < NO)
    n_other = jnp.minimum(other.sum(), NO)
    sell = active & is_sell
    sell_rank = jnp.cumsum(sell) - sell
    keep = keep_other | (sell & (sell_rank < NO - n_other))
    dest = jnp.where(keep, jnp.cumsum(keep) - 1, NO)
    z = jnp.zeros(NO, jnp.int32)
    o_op = z.at[dest].set(_C_OP, mode="drop")
    o_item = z.at[dest].set(_C_ITEM, mode="drop")
    o_n = z.at[dest].set(qty.astype(jnp.int32), mode="drop")
    return o_op, o_item, o_n, jnp.minimum(keep.sum(), NO).astype(jnp.int32)


# ---------------------------------------------------------------------------------------------- unit commands
def unit_commands(raw, op, qc):
    """op int [16] (classes), qc int [16] (quantity class 0..11, before the clamp) -> (u_op, u_arg, u_n, n_cmd, qcmd).
    Clamp as the reference: qc = max(0, min(qc, have - 1)), have = shed stock (PICKUP) / own inventory (PLACE)."""
    n = jnp.minimum(NU, raw.n_units[0])
    valid = jnp.arange(NU) < n
    uop = jnp.where(valid, _OP_UOP[op], 0)
    arg = jnp.where(valid, _OP_ARG[op], 0)
    pick = valid & (op >= K.OP_PICKUP0) & (op < K.OP_PLACE0)
    place = valid & (op >= K.OP_PLACE0)
    have = jnp.where(pick, raw.shed[arg], jnp.where(place, raw.inv[jnp.arange(NU), arg], 0))
    qcmd = jnp.maximum(0, jnp.minimum(qc, have - 1))
    u_n = jnp.where(pick | place, qcmd + 1, 1)
    return uop.astype(jnp.int32), arg.astype(jnp.int32), u_n.astype(jnp.int32), n.astype(jnp.int32), qcmd


def is_qty_op(op):
    return op >= K.OP_PICKUP0


# ---------------------------------------------------------------------------------------------- decide
class Cfg(NamedTuple):
    """greedy: argmax decisions (Kaggle runtime); else ops ~ softmax(masked / T), qty ~ softmax(qty / T), market head
    fires ~ Bernoulli(sigmoid(ks (logit P(order) - logit tau))) (live_rollout / ppo.py).  tau: float or [21]."""
    greedy: bool = True
    T: float = 0.5
    tau: float = 0.5
    ks: float = 4.0


def decide(raw, out, key, cfg: Cfg, mask=None):
    """ONE game: network outputs (op [16,44], qty [16,12], mk [249], v []) -> decisions + log-probs + Act.
    Returns dict(act=Act, op [16] (-1 for absent units), qact [16] (sampled / argmax qty class before the clamp,
    -1 where the op has no quantity), qcmd [16], fire [21] bool, mclass [21] (0 when not fired), umask [16,44],
    lp_u [16], lp_q [16], lp_m [21] (log-probs under the sampling policy; 0 where not applicable), valid [16])."""
    mask = legal_mask(raw) if mask is None else mask
    n = jnp.minimum(NU, raw.n_units[0])
    valid = jnp.arange(NU) < n
    k_op, k_q, k_m = jax.random.split(key, 3)
    zu = jnp.where(mask, out["op"] / cfg.T, NEG)
    lpu_all = jax.nn.log_softmax(zu, -1)
    if cfg.greedy:
        op = jnp.argmax(jnp.where(mask, out["op"], NEG), -1)
    else:
        op = jax.random.categorical(k_op, zu, axis=-1)
    lpq_all = jax.nn.log_softmax(out["qty"] / cfg.T, -1)
    if cfg.greedy:
        qc = jnp.argmax(out["qty"], -1)
    else:
        qc = jax.random.categorical(k_q, out["qty"] / cfg.T, axis=-1)
    hasq = valid & is_qty_op(op)
    qact = jnp.where(hasq, qc, -1)
    u_op, u_arg, u_n, n_cmd, qcmd = unit_commands(raw, op, qc)
    a = jnp.clip(market_fire_logits(out["mk"], cfg.tau, cfg.ks), -LOGIT_CLAMP, LOGIT_CLAMP)   # = logp_terms
    if cfg.greedy:
        fire = market_logit_order(out["mk"]) > _logit(_tau_vec(cfg.tau))
    else:
        fire = jax.random.uniform(k_m, (K.N_HEADS,)) < jax.nn.sigmoid(a)
    mclass = jnp.where(fire, market_class(out["mk"]), 0)
    o_op, o_item, o_n, n_ord = decode_market(mclass)
    lp_u = jnp.where(valid, jnp.take_along_axis(lpu_all, op[:, None], 1)[:, 0], 0.0)
    lp_q = jnp.where(hasq, jnp.take_along_axis(lpq_all, qc[:, None], 1)[:, 0], 0.0)
    lp_m = jnp.where(fire, jax.nn.log_sigmoid(a), jax.nn.log_sigmoid(-a))
    act = Act(u_op, u_arg, u_n, n_cmd, o_op, o_item, o_n, n_ord)
    return dict(act=act, op=jnp.where(valid, op, -1).astype(jnp.int32), qact=qact.astype(jnp.int32), qcmd=qcmd,
                fire=fire, mclass=mclass.astype(jnp.int32), umask=mask, lp_u=lp_u, lp_q=lp_q, lp_m=lp_m, valid=valid)


def policy_step(params, raw, prev_ops, key, team, cfg: Cfg = Cfg(), tabs=None, precision=N.HIGHEST):
    """Batched: RawObs [B,...], prev_ops int [B,16], key, team int [B] -> dict of decide() outputs [B,...] plus
    value [B], the encoder outputs (planes uint8, glob, pos, uf) for the rollout buffer, next_prev [B,16]."""
    tabs = tabs if tabs is not None else E.tables_jnp()
    enc = jax.vmap(E.encode, in_axes=(0, 0, None))(raw, prev_ops, tabs)
    planes, glob, pos, uin = E.net_inputs(enc, tabs)
    out = N.forward(params, planes, glob, pos, uin, team, precision=precision)
    b = planes.shape[0]
    keys = jax.random.split(key, b)
    per = {k: out[k] for k in ("op", "qty", "mk", "v")}
    d = jax.vmap(lambda r, o, k: decide(r, o, k, cfg))(raw, per, keys)
    d["value"] = out["v"]
    d["enc"] = {k: enc[k] for k in ("planes", "glob", "pos", "uf")}
    d["next_prev"] = d["op"]                                   # -1 for absent units, as the reference's shorter list
    return d


policy_step_jit = jax.jit(policy_step, static_argnames=("cfg", "precision"))


# ---------------------------------------------------------------------------------------------- PPO terms
def logp_terms(params, planes_u8, glob, pos, uin, team, umask, uact, qact, mfire, cfg: Cfg, tabs=None,
               precision=N.HIGHEST):
    """ppo.logp_terms in JAX (differentiable w.r.t. params): recorded batch -> dict(lpu [B,16], valid, lpu_all
    [B,16,44], umask, lpq [B,16], qvalid, lpq_all [B,16,12], lpm [B,21], pf [B,21], v [B]).  Market heads use the
    cancellation-free logit (logsumexp(z[1:]) - z0) instead of torch's 1 - softmax[0]; identical up to rounding."""
    tabs = tabs if tabs is not None else E.tables_jnp()
    planes = tabs["q255"][planes_u8.astype(jnp.int32)]
    out = N.forward(params, planes, glob, pos, uin, team, precision=precision)
    zu = jnp.where(umask, out["op"] / cfg.T, NEG)
    lpu_all = jax.nn.log_softmax(zu, -1)
    valid = uact >= 0
    lpu = jnp.take_along_axis(lpu_all, jnp.maximum(uact, 0)[..., None], -1)[..., 0]
    lpq_all = jax.nn.log_softmax(out["qty"] / cfg.T, -1)
    qvalid = qact >= 0
    lpq = jnp.take_along_axis(lpq_all, jnp.maximum(qact, 0)[..., None], -1)[..., 0]
    a = jax.vmap(lambda m: market_fire_logits(m, cfg.tau, cfg.ks))(out["mk"])
    ac = jnp.clip(a, -LOGIT_CLAMP, LOGIT_CLAMP)                # pf clamped to [1e-6, 1 - 1e-6] as ppo.py
    f = mfire.astype(jnp.float32)
    lpm = f * jax.nn.log_sigmoid(ac) + (1 - f) * jax.nn.log_sigmoid(-ac)
    return dict(lpu=lpu, valid=valid, lpu_all=lpu_all, umask=umask, lpq=lpq, qvalid=qvalid, lpq_all=lpq_all,
                lpm=lpm, pf=jax.nn.sigmoid(ac), v=out["v"])


# ---------------------------------------------------------------------------------------------- converters
_UOP_NAMES = list(K.UOP)
_MOP_NAMES = list(K.MOP)


def act_to_dict(a):
    """Act of ONE game -> the Python action dict (farmer / hands / market) of the engine and features.py."""
    a = jax.tree_util.tree_map(np.asarray, a)
    cmds = []
    for i in range(int(a.n_cmd)):
        op, arg, n = int(a.u_op[i]), int(a.u_arg[i]), int(a.u_n[i])
        name = _UOP_NAMES[op]
        if name == "PLANT":
            cmds.append(["PLANT", K.CROPS[arg]])
        elif name in ("PICKUP", "PLACE"):
            cmds.append([name, K.ITEMS[arg], n])
        else:
            cmds.append([name])
    mk = []
    for j in range(int(a.n_ord)):
        op, it, n = _MOP_NAMES[int(a.o_op[j])], int(a.o_item[j]), int(a.o_n[j])
        if op in ("HIRE", "BUY_LAND"):
            mk.append([op])
        elif op == "BUY_SEED":
            mk.append([op, K.CROPS[it], n])
        else:
            mk.append([op, K.ITEMS[it], n])
    return {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk}
