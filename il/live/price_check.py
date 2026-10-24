# Sale timing vs production: in pinned ladder worlds, units sold and average price per unit by product for our seat,
# neural (checkpoint + rule layer) vs a scripted bot.  Usage: price_check.py <ckpt> <scripted> [n_worlds]
import sys, json, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
R = "/Users/asad.ali/kaggriculture"


def one(args):
    who, w = args
    import live_rollout as LR
    from harness import kagsim_trace, mk, PASS
    from rollout import load_world
    d, shops, e0 = load_world(w["path"]); seat = w["seat"]; th = 1 - seat
    g = kagsim_trace.Game(int(d["seed"]), 720, True); g.set_forced_shops(shops)
    if th == 1:
        g.set_weed_prefix(1, e0)
    opp = LR._Tape(d["actions"][th])
    if who.endswith(".pt"):
        from agent import BCAgent
        ag = BCAgent(who, mk_tau=0.6, team=5)
    else:
        ag = mk(who)()
    units = collections.Counter(); rev = collections.Counter(); ou = collections.Counter(); orev = collections.Counter()
    for t in range(720):
        try:
            a = ag(g.observe(seat))
        except Exception:
            a = dict(PASS)
        now = [None, None]; now[seat] = a; now[th] = opp(g.observe(th)); g.step(now[0], now[1])
        for rc in g.trace_step(g.trace_len() - 1).get("receipts", []):
            if rc.get("op") == "SELL" and rc.get("executed"):
                dm = rc["money_after"] - rc["money_before"]
                if rc["seat"] == seat:
                    units[rc["item"]] += rc["executed"]; rev[rc["item"]] += dm
                else:
                    ou[rc["item"]] += rc["executed"]; orev[rc["item"]] += dm
        if g.done:
            break
    return who, w["ep"], dict(units), dict(rev), dict(ou), dict(orev), g.reward(seat) - g.reward(th)


if __name__ == "__main__":
    ck, scr = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    N = {r["ep"]: r for r in json.load(open(f"{R}/il/logs/ladder_rl_v3d_it30_t5_mnfh_test.json"))}
    S = {r["ep"]: r for r in json.load(open(f"{R}/il/logs/ladder_main.py_test.json"))}
    W = {w["ep"]: w for w in json.load(open(f"{R}/il/live/ladder_worlds.json"))}
    eps = [e for e in sorted(N) if N[e]["margin"] < 0 and S[e]["margin"] > 0][:n]
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.map(one, [(ck, W[e]) for e in eps] + [(scr, W[e]) for e in eps])
    agg = {}
    for who, ep, u, r, ou, orv, m in res:
        A = agg.setdefault("neural" if who.endswith(".pt") else "scripted", dict(u=collections.Counter(), r=collections.Counter(), ou=collections.Counter(), orv=collections.Counter(), m=[]))
        A["u"].update(u); A["r"].update(r); A["ou"].update(ou); A["orv"].update(orv); A["m"].append(m)
    k = len(eps)
    print(f"{k} held-out ladder worlds where the neural bot (with rules) loses and STACK1_RACE4 wins")
    for name, A in agg.items():
        print(f"  {name}: mean margin {np.mean(A['m']):+,.0f} | our sales ${sum(A['r'].values())/k:,.0f}/game | rival sales ${sum(A['orv'].values())/k:,.0f}/game")
    print("  per product: units/game and $/unit, neural vs scripted (rival's $/unit in brackets)")
    for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"):
        a, b = agg["neural"], agg["scripted"]
        f = lambda A, key_u, key_r: (A[key_u][p] / k, A[key_r][p] / max(1, A[key_u][p]))
        (nu, np_), (su, sp), (_, onp), (_, osp) = f(a, "u", "r"), f(b, "u", "r"), f(a, "ou", "orv"), f(b, "ou", "orv")
        print(f"   {p:10s} neural {nu:5.1f}u @ ${np_:6.1f} [rival {onp:6.1f}] | scripted {su:5.1f}u @ ${sp:6.1f} [rival {osp:6.1f}]")
