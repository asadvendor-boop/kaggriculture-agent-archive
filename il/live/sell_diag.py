"""Why does SELL_ORDER=base move our own bank so much? Plays N worlds of a benchmark set in-process with the current env
rules and counts, per game: market turns, turns whose decoded list exceeded the 10-order cap (and which SELLs the cap
cut), phantom SELLs (nothing held) and the slot of the first premium SELL; plus our sold units and revenue per product
at the end (from the engine's own counters when present).  Usage: SELL_ORDER=base python live/sell_diag.py ckpt/x.pt [n] [DAY]"""
import sys, os, json, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import features as F
import live_rollout as LR

ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3; day = sys.argv[3] if len(sys.argv) > 3 else "sevloss"
W = json.load(open(f"/Users/asad.ali/kaggriculture/il/live/top_worlds_{day}_300.json"))[:n]
_orig = F.decode_market
st = collections.Counter(); cut_items = collections.Counter(); first_slot = []


def _spy(classes, obs=None):
    out = _orig(classes, obs)
    st["turns"] += 1
    # rebuild the uncapped list the same way decode_market does (before its cap)
    full = []
    for p in ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]:
        k = int(classes[F.HEAD_INDEX["SELL:" + p]])
        if k > 0:
            full.append(["SELL", p])
    nonsell = int(classes[F.HEAD_INDEX["HIRE"]]) + int(classes[F.HEAD_INDEX["BUY_LAND"]]) + \
        sum(int(classes[F.HEAD_INDEX["BUY_ANIMAL:" + a]]) > 0 for a in F.ANIMALS) + \
        sum(int(classes[F.HEAD_INDEX["BUY_SEED:" + c]]) > 0 for c in F.CROPS) + \
        (int(classes[F.HEAD_INDEX["BUY_PRODUCT:WHEAT"]]) > 0) + (int(classes[F.HEAD_INDEX["BUY_PRODUCT:FERTILIZER"]]) > 0)
    if len(full) + nonsell > 10:
        st["capped_turns"] += 1
        kept = {o[1] for o in out if o and o[0] == "SELL"}
        for o in full:
            if o[1] not in kept:
                cut_items[o[1]] += 1
    if obs is not None:
        priv = obs.get("private") or {}; held = collections.Counter({k: int(v) for k, v in (priv.get("shed") or {}).items()})
        for inv in priv.get("inventories") or []:
            for k, v in (inv or {}).items():
                held[k] += int(v)
        for i, o in enumerate(out):
            if o and o[0] == "SELL":
                st["sells"] += 1
                if held.get(o[1], 0) <= 0:
                    st["phantom_sells"] += 1
                    if i < 3:
                        st["phantom_in_slot_0-2"] += 1
        prem = [i for i, o in enumerate(out) if o and o[0] == "SELL" and o[1] in ("MELON", "WOOL", "MILK") and held.get(o[1], 0) > 0]
        if prem:
            first_slot.append(prem[0])
    return out


F.decode_market = _spy
for w in W:
    job = dict(weights=ck, team=int(os.environ.get("TEAM", "10")), tau=float(os.environ.get("TAU", "0.5")), T=0.2, ks=10.0,
               greedy=True, seed=0, games=[(0, w["seat"], "TAPE:" + w["path"])])
    r = LR._run_batch(job)[0]
    print(f"ep {w['ep']} seat {w['seat']} vs {w['opp']}: margin {r['margin']:+,.0f} bank {r['bank']:,.0f} opp {r['opp_bank']:,.0f}", flush=True)
print(f"SELL_ORDER={F.SELL_ORDER}: {dict(st)} | cap cut SELLs by item {dict(cut_items)} | "
      f"mean slot of the first held premium SELL {sum(first_slot) / max(1, len(first_slot)):.2f}", flush=True)
