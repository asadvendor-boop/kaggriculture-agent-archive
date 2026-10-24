"""Numpy-only runtime of the imitation/RL policy (no torch): forward pass + decoding identical to agent.BCAgent.
Embedded verbatim into the Kaggle main.py by export.py, together with features.py and the weights blob."""
import numpy as np

_MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _conv3(x, w, b):
    """x [C,10,10], w [O,C,3,3], b [O] -> [O,10,10] (padding 1) via im2col."""
    C = x.shape[0]
    xp = np.zeros((C, 12, 12), np.float32); xp[:, 1:11, 1:11] = x
    cols = np.empty((C, 3, 3, 10, 10), np.float32)
    for dy in range(3):
        for dx in range(3):
            cols[:, dy, dx] = xp[:, dy:dy + 10, dx:dx + 10]
    out = w.reshape(w.shape[0], -1) @ cols.reshape(C * 9, 100)
    return (out + b[:, None]).reshape(-1, 10, 10)


def _relu(x):
    return np.maximum(x, 0, out=x)


class NumpyNet:
    def __init__(self, P, width, blocks, n_teams):
        self.P = {k: v.astype(np.float32) for k, v in P.items()}
        self.width, self.blocks, self.n_teams = width, blocks, n_teams
        yy, xx = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        self.coords = np.stack([xx / 9.0, yy / 9.0]).astype(np.float32)

    def lin(self, name, x):
        return x @ self.P[name + ".weight"].T + self.P[name + ".bias"]

    def forward(self, planes, glob, pos, uin, team=0):
        """planes [C,10,10] float 0..1, glob [G], pos [n,2], uin [n,U] -> op [n,44], qty [n,12], market [sum]."""
        P = self.P
        if self.n_teams:
            glob = np.concatenate([glob, P["team.weight"][team]])
        g = _relu(self.lin("g2", _relu(self.lin("g1", glob))))
        x = np.concatenate([planes, self.coords], 0)
        x = _relu(_conv3(x, P["stem.weight"], P["stem.bias"]) + self.lin("film", g)[:, None, None])
        for i in range(self.blocks):
            h = _relu(_conv3(x, P[f"blocks.{i}.c1.weight"], P[f"blocks.{i}.c1.bias"]))
            x = _relu(x + _conv3(h, P[f"blocks.{i}.c2.weight"], P[f"blocks.{i}.c2.bias"]))
        flat = x.reshape(self.width, 100)
        idx = np.clip(pos[:, 1] * 10 + pos[:, 0], 0, 99)
        cell = flat[:, idx].T                                            # [n,W]
        u = np.concatenate([cell, np.repeat(g[None], len(idx), 0), uin], 1)
        h = _relu(self.lin("u2", _relu(self.lin("u1", u))))
        pooled = np.concatenate([flat.mean(1), flat.max(1), g])
        hm = _relu(self.lin("m2", _relu(self.lin("m1", pooled))))
        self.hm = hm
        return self.lin("uop", h), self.lin("uqty", h), self.lin("mk", hm)

    def ar_greedy(self, hm, tau):
        """Order-by-order market head (numpy port of model.Net.ar_greedy): [(op, item, qty)] until the first STOP."""
        P = self.P; H = hm.shape[0]; out = []
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        h = np.tanh(self.lin("ar_h0", hm))
        for k in range(10):
            lo = self.lin("ar_op", np.concatenate([h, P["ar_slot_emb.weight"][k]]))
            p = np.exp(lo - lo.max()); p /= p.sum()
            if not (1.0 - p[0] > tau):
                break
            op = int(1 + np.argmax(p[1:])); oe = P["ar_op_emb.weight"][op]
            has = 2 <= op <= 5                     # SELL / BUY_*: item + qty; STOP / EMPTY / HIRE / BUY_LAND: 0, 0 as in training
            it = int(np.argmax(self.lin("ar_item", np.concatenate([h, oe])) + P["ar_item_bias"][op])) if has else 0; ie = P["ar_item_emb.weight"][it]
            q = int(np.argmax(self.lin("ar_qty", np.concatenate([h, oe, ie])))) if has else 0; qe = P["ar_qty_emb.weight"][q]
            out.append((op, it, q))
            x = np.concatenate([oe, ie, qe]); Hh = h.shape[0]
            gi = P["ar_cell.weight_ih"] @ x + P["ar_cell.bias_ih"]; gh = P["ar_cell.weight_hh"] @ h + P["ar_cell.bias_hh"]
            r = sig(gi[:Hh] + gh[:Hh]); z = sig(gi[Hh:2 * Hh] + gh[Hh:2 * Hh]); n = np.tanh(gi[2 * Hh:] + r * gh[2 * Hh:])
            h = (1.0 - z) * n + z * h
        return out


def legal_mask_np(obs, k, pos, F):
    me = int(obs["player"]); farm = obs["farms"][me]; priv = obs["private"]
    x, y = int(pos[0]), int(pos[1])
    t = farm["tiles"][y][x]
    invs = priv.get("inventories") or [{}]
    iv = invs[k] if k < len(invs) else {}
    shed = priv.get("shed", {}) or {}; seeds = priv.get("seeds", {}) or {}
    day = int(obs["step"]) // 24
    at_shed = (x, y) in F.SHED_ACCESS
    m = np.zeros(F.N_OPS, bool)
    def ok(name):
        m[F.OP_INDEX[name]] = True
    ok("PASS")
    for mv, (dx, dy) in _MOVES.items():
        if 0 <= x + dx < 10 and 0 <= y + dy < 10:
            ok(mv)
    is_d = isinstance(t, dict)
    kind = t.get("kind") if is_d else t
    animal = is_d and "animal" in t
    if kind == "PLANT":
        if not t.get("watered_today"):
            ok("WATER")
        ci = F.CROP_INFO.get(t.get("crop"))
        if ci and int(t.get("yield_units", 0)) > 0 and day - int(t.get("planted_day", day)) >= ci[0]:
            ok("HARVEST")
        if iv.get("FERTILIZER", 0) > 0:
            ok("FERTILIZE")
    if animal:
        if int(t.get("yield_units", 0)) > 0:
            ok("HARVEST")
        if not t.get("fed_today") and iv.get("WHEAT", 0) > 0:
            ok("FEED")
        if t.get("fertilizer_available"):
            ok("COLLECT_FERTILIZER")
        if not t.get("cared_today"):
            ok("CARE")
    if t is not None and t != "LOCKED" and not animal:
        ok("DIG")
    if t is None:
        ok("BUILD_COOP"); ok("BUILD_PASTURE")
        for c in F.CROPS:
            if seeds.get(c, 0) > 0:
                ok("PLANT_" + c)
    if at_shed:
        if any(v > 0 for v in iv.values()):
            ok("DROP")
        for it in F.ITEMS:
            if shed.get(it, 0) > 0:
                ok("PICKUP_" + it)
    for it in F.ITEMS:
        if iv.get(it, 0) > 0:
            if it in F.ANIMALS:
                if is_d and not animal and kind == {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}[it]:
                    ok("PLACE_" + it)
                elif at_shed:
                    ok("PLACE_" + it)
            elif at_shed:
                ok("PLACE_" + it)
    return m


class NumpyAgent:
    def __init__(self, net, F, tau=0.35, team=0):
        self.net, self.F, self.tau, self.team = net, F, tau, team
        self.state = {}
        self.MK = getattr(F, "F8", None) or F           # extended fixed menu (features8) when export.py attached it
        self.off = self.MK.MARKET_OFF if self.MK is not F else np.cumsum([0] + [n for _, n in F.MARKET_HEADS])

    def __call__(self, obs, cfg=None):
        F = self.F
        me, step = int(obs["player"]), int(obs["step"])
        st = self.state.get(me)
        if st is None or step == 0 or step <= st["step"]:
            st = self.state[me] = {"step": -1, "prev": None}
        st["step"] = step
        planes, g, pos, uf = F.encode_obs(obs, st["prev"])
        lop, lq, lm = self.net.forward(planes.astype(np.float32) / 255.0, g, pos.astype(np.int64), uf, self.team)
        priv = obs["private"]; shed = priv.get("shed", {}) or {}; invs = priv.get("inventories") or [{}]
        cmds, ops = [], []
        for k in range(len(pos)):
            allowed = legal_mask_np(obs, k, pos[k], F)
            oc = int(np.argmax(np.where(allowed, lop[k], -1e9)))
            name = F.OPS[oc]; qc = 0
            if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                qc = int(np.argmax(lq[k])); it = name.split("_", 1)[1]
                have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                qc = max(0, min(qc, int(have) - 1))
            cmds.append(F.unit_command(oc, qc)); ops.append(oc)
        st["prev"] = ops
        classes = []
        MK = self.MK
        for h in range(len(MK.MARKET_HEADS)):
            z = lm[self.off[h]:self.off[h + 1]].astype(np.float64); p = np.exp(z - z.max()); p /= p.sum()
            th = self.tau.get(MK.MARKET_HEADS[h][0], self.tau.get("_default", 0.6)) if isinstance(self.tau, dict) else self.tau
            classes.append(int(1 + np.argmax(p[1:])) if 1.0 - p[0] > th else 0)
        FA = getattr(F, "FA", None)
        if FA is not None and "ar_op.weight" in self.net.P:           # order-by-order market head
            th = self.tau.get("_default", 0.5) if isinstance(self.tau, dict) else self.tau
            seq = self.net.ar_greedy(self.net.hm, th)
            mk = FA.decode_seq([o for o, _, _ in seq], [i for _, i, _ in seq], [q for _, _, q in seq])
        else:
            mk = MK.decode_market(classes, obs)
        act = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk}
        return F.midnight_fix(obs, act) if hasattr(F, "midnight_fix") else act
