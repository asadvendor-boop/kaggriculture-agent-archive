"""Smoke test of the rule layer's switchable plan rules (Sep 28: DSM's Sep 26 plan = LAND2_DAY / NO_LAND3 / EARLY_COWS /
MELON_WAVE): plays a few leader-test worlds in-process with the given env rules and prints, per game, what the rules
ordered and what the farm shows at the end of each day (extra lands, cows placed + in the shed, melon tiles planted on
day >= MELON_DAY, cash) plus the final margin.  Usage: RULES_ENV="MELON_WAVE=4 EARLY_COWS=3" python live/rule_smoke.py
ckpt/x.pt [n_worlds] [DAY]"""
import sys, os, json, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
for kv in os.environ.get("RULES_ENV", "").split():
    k, v = kv.split("="); os.environ[k] = v
import features as F
import live_rollout as LR

ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3; day = sys.argv[3] if len(sys.argv) > 3 else "fresh"
W = json.load(open(f"/Users/asad.ali/kaggriculture/il/live/top_worlds_{day}_300.json"))
if os.environ.get("EPS"):                             # Sep 28: pick worlds by episode id (e.g. the 2-yarn-store live losses)
    want = [int(x) for x in os.environ["EPS"].split(",")]; W = [w for e in want for w in W if int(w["ep"]) == e]
W = W[:n]
fired = collections.defaultdict(list); daily = collections.defaultdict(list)
_orig = F._plan_fix


def _spy(obs, action):
    out = _orig(obs, action)
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"]); d, h = divmod(step, 24)
    key = (int(obs.get("_seed_tag", 0)), me)
    ua = lambda a: [a.get("farmer")] + list(a.get("hands") or [])
    diff_u = [(i, a, b) for i, (a, b) in enumerate(zip(ua(action), ua(out))) if list(a or []) != list(b or [])]
    ma = [list(o) for o in (action.get("market") or [])]; mb = [list(o) for o in (out.get("market") or [])]
    if diff_u or ma != mb:
        add = [o for o in mb if o not in ma]; rem = [o for o in ma if o not in mb]
        _spy.log.append(f"d{d:02d}h{h:02d} units {[(i, list(b)) for i, a, b in diff_u]} market +{add} -{rem}")
    if h == 23:
        priv = obs.get("private") or {}
        cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        late = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "MELON"
                   and int(t.get("planted_day", -1)) >= F.MELON_DAY)
        sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        rs = sum(1 for row in obs["farms"][1 - me]["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        yarn = sum(x == "YARN_STORE" for x in ((obs.get("town") or {}).get("unlocked_shops") or []))
        _spy.days.append(f"d{d:02d}: lands {len(farm.get('unlocked_quadrants') or []) - 1} cows {cows}+{int((priv.get('shed') or {}).get('COW', 0))}"
                         f" sheep {sheep}+{int((priv.get('shed') or {}).get('SHEEP', 0))} (rival {rs}, yarn {yarn}, wool ${float(obs['market']['prices'].get('WOOL', 0)):.0f})"
                         f" tomato {sum(1 for row in farm['tiles'] for t in row if isinstance(t, dict) and t.get('crop') == 'TOMATO')}"
                         f" lateMelon {late} ${float(farm['money']):,.0f}")
    return out


_spy.log = []; _spy.days = []
F._plan_fix = _spy
flags = {k: getattr(F, k, None) for k in ("SE_GUARD", "SE_SHOPS", "SHEEP_YARN", "TOMATO_SHOPS", "COW_MILK", "FEED_GUARD")}
for w in W:
    _spy.log = []; _spy.days = []
    job = dict(weights=ck, team=int(os.environ.get("TEAM", "10")), tau=float(os.environ.get("TAU", "0.5")), T=0.2, ks=10.0,
               greedy=True, seed=0, games=[(0, w["seat"], "TAPE:" + w["path"])])
    r = LR._run_batch(job)[0]
    print(f"== ep {w['ep']} seat {w['seat']} vs {w['opp']}: margin {r['margin']:+,.0f} bank {r['bank']:,.0f}  flags {dict((k, getattr(F, k)) for k in flags)}")
    for s in _spy.log[:40]:
        print("   ", s)
    for s in _spy.days[:25]:
        print("      ", s)
    sys.stdout.flush()
