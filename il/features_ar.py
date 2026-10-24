"""Order-by-order market vocabulary (the "full" output menu): a turn's market orders are a sequence of up to 10 slots,
each (op, item, quantity).  Covers everything the leaders send: EMPTY dead slots used for positional padding ([] or
["INVALID"]), repeated orders for the same item, and large quantities (a BUY bigger than the cash allows fills as much
as is affordable, so "buy max" is just a big number).

  seq_labels(market)            -> int16 [10, 3] (op, item, qty) with STOP after the last order
  decode_seq(ops, items, qtys)  -> market order list (EMPTY -> [])
"""
import numpy as np
from features import PRODUCTS, ANIMALS, CROPS

AR_OPS = ["STOP", "EMPTY", "SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "HIRE", "BUY_LAND"]
AR_OP = {o: i for i, o in enumerate(AR_OPS)}
AR_ITEMS = PRODUCTS + ANIMALS                                  # 12 items
AR_ITEM = {o: i for i, o in enumerate(AR_ITEMS)}
AR_QTY = list(range(1, 41)) + [45, 50, 60, 70, 80, 100, 150, 200, 1000]     # every quantity up to 40 exact
SLOTS = 10
HAS_ITEM = {"SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}
HAS_QTY = {"SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}
VALID_ITEMS = {"SELL": PRODUCTS, "BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS, "BUY_PRODUCT": ["WHEAT", "FERTILIZER"]}
ITEM_MASK = np.zeros((len(AR_OPS), len(AR_ITEMS)), bool)         # [op, item] -> allowed
for _o, _its in VALID_ITEMS.items():
    for _it in _its:
        ITEM_MASK[AR_OP[_o], AR_ITEM[_it]] = True
_QLOG = np.log(np.array(AR_QTY, np.float64))


def qty_class(q):
    q = max(1, int(q))
    return int(np.argmin(np.abs(_QLOG - np.log(min(q, 1000)))))


def parse_order(o):
    """(op, item, qty_class) of one raw order; malformed / padding entries -> EMPTY."""
    if not isinstance(o, (list, tuple)) or not o or o[0] not in AR_OP or o[0] in ("STOP", "EMPTY"):
        return AR_OP["EMPTY"], 0, 0
    op = o[0]
    if op in ("HIRE", "BUY_LAND"):
        return AR_OP[op], 0, 0
    item = o[1] if len(o) > 1 else None
    if item not in VALID_ITEMS.get(op, ()):
        return AR_OP["EMPTY"], 0, 0
    try:
        q = int(float(o[2])) if len(o) > 2 else 1
    except Exception:
        return AR_OP["EMPTY"], 0, 0
    if q <= 0:
        return AR_OP["EMPTY"], 0, 0
    return AR_OP[op], AR_ITEM[item], qty_class(q)


def seq_labels(market):
    lab = np.zeros((SLOTS, 3), np.int16)                          # all STOP
    orders = list(market or [])[:SLOTS]
    for k, o in enumerate(orders):
        lab[k] = parse_order(o)
    return lab


def decode_seq(ops, items, qtys):
    out = []
    for op, it, q in zip(ops, items, qtys):
        name = AR_OPS[int(op)]
        if name == "STOP":
            break
        if name == "EMPTY":
            out.append([])
        elif name in ("HIRE", "BUY_LAND"):
            out.append([name])
        else:
            out.append([name, AR_ITEMS[int(it)], AR_QTY[int(q)]])
    return out[:SLOTS]
