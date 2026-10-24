"""Literal pure-Python port of sim_trace.hpp process_market / commit_unit / do_hire / do_buy_land (the per-unit
lockstep loop, one unit at a time). Used only as a test oracle for the closed-form JAX market phase."""
try:
    from . import consts as C
except ImportError:
    import consts as C


def fib(n):
    return int(C.FIB[n])


def process_market(st, orders):
    """st: dict with money[2], shed[2][12], seeds[2][5], minv[9], hires[2], n_units[2], ux[2][U], uy[2][U],
    n_quads[2] (mutated in place). orders[p] = list of (op, item, n) (already converted; NONE = dead)."""
    nq = [min(len(orders[p]), C.NSLOT) for p in range(2)]
    for i in range(max(nq)):
        os_ = [None, None]
        for p in range(2):
            if i < nq[p]:
                op, item, n = orders[p][i]
                if op in (C.M_HIRE, C.M_BUY_LAND):
                    os_[p] = {"type": op, "item": 0, "rem": 1, "live": True}
                elif op != C.M_NONE and n > 0:
                    os_[p] = {"type": op, "item": item, "rem": n, "live": True}
        for p in range(2):
            o = os_[p]
            if o is None or not o["live"]:
                continue
            if o["type"] == C.M_HIRE:
                do_hire(st, p)
                o["live"] = False
            elif o["type"] == C.M_BUY_LAND:
                do_buy_land(st, p)
                o["live"] = False
        while True:
            q = [None, None]
            for p in range(2):
                o = os_[p]
                if o is None or not o["live"] or o["rem"] <= 0:
                    continue
                t, it = o["type"], o["item"]
                if t == C.M_SELL and 0 <= it < C.N_PRODUCTS:
                    q[p] = (t, it, C.market_price(it, st["minv"][it]))
                elif t == C.M_BUY_PRODUCT and it in (C.WHEAT, C.FERTILIZER):
                    q[p] = (t, it, C.market_price(it, st["minv"][it] - 1))
                elif t == C.M_BUY_SEED and 0 <= it < C.N_CROPS:
                    q[p] = (t, it, int(C.CROP_SEED[it]))
                elif t == C.M_BUY_ANIMAL and C.GOOSE <= it <= C.SHEEP:
                    q[p] = (t, it, int(C.ANIMAL_COST[it - C.GOOSE]))
                else:
                    o["live"] = False
            if q[0] is None and q[1] is None:
                break
            committed = False
            for p in range(2):
                if q[p] is None:
                    continue
                if commit_unit(st, p, *q[p]):
                    os_[p]["rem"] -= 1
                    committed = True
                else:
                    os_[p]["live"] = False
            if not committed:
                break


def commit_unit(st, p, op, item, price):
    shed = st["shed"][p]
    if op == C.M_SELL:
        if shed[item] <= 0:
            return False
        shed[item] -= 1
        st["money"][p] += price
        if price > 1:
            st["minv"][item] += 1
        return True
    if op == C.M_BUY_PRODUCT:
        if st["money"][p] < price or sum(shed) >= C.SHED_CAP:
            return False
        st["money"][p] -= price
        shed[item] += 1
        st["minv"][item] -= 1
        return True
    if op == C.M_BUY_SEED:
        if st["money"][p] < price:
            return False
        st["money"][p] -= price
        st["seeds"][p][item] += 1
        return True
    if op == C.M_BUY_ANIMAL:
        if st["money"][p] < price or sum(shed) >= C.SHED_CAP:
            return False
        st["money"][p] -= price
        shed[item] += 1
        return True
    return False


def do_hire(st, p):
    cost = fib(st["hires"][p])
    if st["money"][p] < cost or st["n_units"][p] >= C.U:
        return
    st["money"][p] -= cost
    st["hires"][p] += 1
    occ = [0, 0, 0, 0]
    for u in range(st["n_units"][p]):
        for k, (ax, ay) in enumerate(C.SHED_ACCESS):
            if st["ux"][p][u] == ax and st["uy"][p][u] == ay:
                occ[k] += 1
    best = 0
    for k in range(1, 4):
        if occ[k] < occ[best]:
            best = k
    idx = st["n_units"][p]
    st["n_units"][p] += 1
    st["ux"][p][idx], st["uy"][p][idx] = C.SHED_ACCESS[best]


def do_buy_land(st, p):
    extra = st["n_quads"][p] - 1
    if extra >= 3:
        return
    cost = int(C.LAND_PRICES[extra])
    if st["money"][p] < cost:
        return
    st["money"][p] -= cost
    st["n_quads"][p] += 1
