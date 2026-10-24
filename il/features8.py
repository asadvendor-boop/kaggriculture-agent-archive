"""Extended fixed market menu ("menu 8"): the old per-(op, item) quantity heads of features.py with the quantity ranges
the top-10 leaders actually use (study of 150 seats per team, Sep 24 2026) plus two padding heads.

  SELL:<product>          0 none, 1..40 exact, 41 = ALL (sends 1000: the engine sells what the shed holds)
  BUY_SEED:<crop>         0..20 exact, then 25 30 40 60 100
  BUY_PRODUCT:WHEAT       0..40 exact, then 50 70 100
  BUY_PRODUCT:FERTILIZER  0..20 exact, then 25 30 40
  BUY_ANIMAL:<animal>     0..10 exact, then 15 25 50 100 200 ("buy as many as affordable" orders fill partially)
  HIRE                    0..10
  BUY_LAND                0..3
  PAD_LEAD                0..5 dead slots before the first order (THIRD FARM CLUB, mtmr_s1, SpaTaro, Boey)
  PAD_BUY                 0..5 dead slots between the sells/hires and the first purchase (SpaTaro, mtmr_s1)

Dead slots only matter when a real order follows them (the lockstep market settles slot k of both farms together, so
a dead slot pushes the later orders behind the rival's); trailing dead slots and pad-only turns are dropped.

  labels_from_seq(seq)          int8 [10, 3] order-by-order labels (features_ar.seq_labels) -> int16 [N_HEADS] classes
  decode_market(classes, obs)   classes -> market order list (canonical order, pads, at most 10 slots)
"""
import numpy as np
from features import PRODUCTS, CROPS, ANIMALS, BASE
import features_ar as FA

_R = lambda a, b: list(range(a, b + 1))
QVALS = {"SELL": _R(0, 40) + [1000], "BUY_SEED": _R(0, 20) + [25, 30, 40, 60, 100],
         "BUY_PRODUCT:WHEAT": _R(0, 40) + [50, 70, 100], "BUY_PRODUCT:FERTILIZER": _R(0, 20) + [25, 30, 40],
         "BUY_ANIMAL": _R(0, 10) + [15, 25, 50, 100, 200], "HIRE": _R(0, 10), "BUY_LAND": _R(0, 3),
         "PAD_LEAD": _R(0, 5), "PAD_BUY": _R(0, 5)}
MARKET_HEADS = ([("SELL:" + p, QVALS["SELL"]) for p in PRODUCTS] + [("BUY_SEED:" + c, QVALS["BUY_SEED"]) for c in CROPS] +
                [("BUY_PRODUCT:WHEAT", QVALS["BUY_PRODUCT:WHEAT"]), ("BUY_PRODUCT:FERTILIZER", QVALS["BUY_PRODUCT:FERTILIZER"])] +
                [("BUY_ANIMAL:" + a, QVALS["BUY_ANIMAL"]) for a in ANIMALS] +
                [("HIRE", QVALS["HIRE"]), ("BUY_LAND", QVALS["BUY_LAND"]), ("PAD_LEAD", QVALS["PAD_LEAD"]), ("PAD_BUY", QVALS["PAD_BUY"])])
HEAD_INDEX = {h: i for i, (h, _) in enumerate(MARKET_HEADS)}
HEAD_SIZES = [len(v) for _, v in MARKET_HEADS]
N_HEADS = len(MARKET_HEADS)                                   # 23
MARKET_OFF = np.concatenate([[0], np.cumsum(HEAD_SIZES)]).astype(int)
N_CLASSES = int(MARKET_OFF[-1])
SELL_ORDER = ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]   # leaders' canonical order
BUY_OPS = ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT")


def value_class(vals, q):
    """Nearest class of quantity q (exact where the menu is exact, log-nearest above; SELL above 40 -> ALL)."""
    q = int(q)
    if q <= 0:
        return 0
    if vals[-1] == 1000:                                        # SELL: 1..40 exact, anything larger = sell all
        return q if q <= 40 else len(vals) - 1
    if q >= vals[-1]:
        return len(vals) - 1
    v = np.array(vals[1:], np.float64)
    return 1 + int(np.argmin(np.abs(np.log(v) - np.log(q))))


def labels_from_seq(seq):
    """seq int [10, 3] (features_ar labels: op, item, qty class; STOP ends) -> int16 [N_HEADS] menu-8 classes."""
    lab = np.zeros(N_HEADS, np.int16)
    ops = [FA.AR_OPS[int(o)] for o in seq[:, 0]]
    n = ops.index("STOP") if "STOP" in ops else len(ops)
    tot = {}; first_real = None; pad_run = 0; pad_buy = 0; seen_nonbuy = False; buy_seen = False
    for k in range(n):
        op = ops[k]
        if op == "EMPTY":
            pad_run += 1
            continue
        if first_real is None:
            first_real = k
        if op in BUY_OPS and not buy_seen:
            buy_seen = True
            if seen_nonbuy:
                pad_buy = pad_run                               # dead slots right before the first purchase
        if op not in BUY_OPS:
            seen_nonbuy = True
        pad_run = 0
        if op == "HIRE":
            key = "HIRE"; q = 1
        elif op == "BUY_LAND":
            key = "BUY_LAND"; q = 1
        else:
            item = FA.AR_ITEMS[int(seq[k, 1])]; q = FA.AR_QTY[int(seq[k, 2])]
            key = op + ":" + item
        tot[key] = tot.get(key, 0) + q
    for key, q in tot.items():
        h = HEAD_INDEX.get(key)
        if h is None:
            continue
        lab[h] = value_class(MARKET_HEADS[h][1], q)
    if first_real is not None:
        lab[HEAD_INDEX["PAD_LEAD"]] = min(first_real, 5)
        lab[HEAD_INDEX["PAD_BUY"]] = min(pad_buy, 5)
    return lab


def decode_market(classes, obs=None):
    """[N_HEADS] class ints -> market order list: pads, sells, hires, land, animals, seeds, product buys (the leaders'
    canonical order), pads before the first purchase; at most 10 slots (pads go first, then trailing sells)."""
    c = [int(x) for x in classes]
    val = lambda name: MARKET_HEADS[HEAD_INDEX[name]][1][c[HEAD_INDEX[name]]]
    sells = [["SELL", p, val("SELL:" + p)] for p in SELL_ORDER if c[HEAD_INDEX["SELL:" + p]] > 0]
    mid = [["HIRE"]] * val("HIRE") + [["BUY_LAND"]] * val("BUY_LAND")
    buys = [["BUY_ANIMAL", a, val("BUY_ANIMAL:" + a)] for a in ANIMALS if c[HEAD_INDEX["BUY_ANIMAL:" + a]] > 0]
    buys += [["BUY_SEED", s, val("BUY_SEED:" + s)] for s in CROPS if c[HEAD_INDEX["BUY_SEED:" + s]] > 0]
    buys += [["BUY_PRODUCT", p, val("BUY_PRODUCT:" + p)] for p in ("WHEAT", "FERTILIZER") if c[HEAD_INDEX["BUY_PRODUCT:" + p]] > 0]
    real = sells + mid + buys
    if not real:
        return []
    lead = val("PAD_LEAD"); pb = val("PAD_BUY") if (buys and (sells or mid)) else 0
    while len(real) + lead + pb > 10 and (lead or pb):          # pads are the first to go
        if pb:
            pb -= 1
        else:
            lead -= 1
    if len(real) > 10:                                          # then trailing sells (they can wait an hour)
        other = (mid + buys)[:10]
        sells = sells[:10 - len(other)]
        real = sells + other
        mid = [o for o in other if o[0] in ("HIRE", "BUY_LAND")]; buys = [o for o in other if o[0] in BUY_OPS]
    head = sells + mid
    return [[]] * lead + head + [[]] * pb + buys


def roundtrip_ok(seq):
    """True when decode(labels(seq)) reproduces seq's orders exactly (same merged quantities, same pads before real
    orders); order within the canonical blocks is not compared."""
    lab = labels_from_seq(seq)
    dec = decode_market(lab)
    ops = [FA.AR_OPS[int(o)] for o in seq[:, 0]]
    n = ops.index("STOP") if "STOP" in ops else len(ops)
    rec = FA.decode_seq(seq[:n, 0], seq[:n, 1], seq[:n, 2])
    while rec and rec[-1] == []:
        rec.pop()
    return _merged(rec) == _merged(dec) and _lead(rec) == _lead(dec)


def _merged(L):
    agg = {}
    for o in L:
        if not o:
            continue
        k = (o[0], o[1] if len(o) > 2 else "")
        q = o[2] if len(o) > 2 else 1
        agg[k] = agg.get(k, 0) + q
    return sorted((k[0], k[1], min(v, 41) if k[0] == "SELL" and v > 40 else v) for k, v in agg.items())


def _lead(L):
    n = 0
    for o in L:
        if o:
            break
        n += 1
    return n if any(L) else 0
