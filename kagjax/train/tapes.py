"""Recorded leader games as non-reactive opponents (ppo_live's TOP worlds, in JAX).

A tape world replays a recorded Kaggle game (A's engine with the seed's end-of-day words, rng.batch_words): the leader's
seat plays its recorded actions, the learner plays the other seat, and the world is PINNED to the recording exactly as
the C++ reference does it (tools/topruler.run_tape: set_forced_shops(recorded shops); set_weed_prefix(1, e0) when the
leader is seat 1): the engine's end-of-day stream is shared by both farms and the shop draw, so without pinning a
learner farm that differs from the recorded player's shifts the leader's weeds and the shops (audit, Sep 26).
RealBackend.step_tapes implements the pinning by rebuilding the day's words from the engine's own empty-tile counts.  Episodes that belong to any
benchmark (il/live/*heldout*, *top_worlds*, *ladder* json files: the leader tests and the ladder test) are never
used; of the rest, episodes with ep % 10 == 0 are held out for the JAX-side TOP evaluation.

build_bank(...) -> dict of numpy arrays (put on the device by the trainer):
  words  uint32 [T,30,624]   end-of-day words of the tape's seed
  uop/uarg int16, un int16 [T,720,40]; mtype/mitem int16, mn int32 [T,720,10]   the leader's recorded actions
                             (A's state.encode_action: exactly the C++ conv_action parsing)
  lseat  int32 [T]           the learner's seat (= 1 - leader seat)
  shops_rec int32 [T,8]      the recorded shop unlock sequence (-1 = none)      } from a C++ replay of both recorded
  e0_rec int32 [T,30]        recorded farm-0 empty tiles per day end            } seats (topruler.recorded_world)
  meta   list of (ep, leader team, leader seat, recorded leader reward, recorded learner-seat reward)
"""
import os, sys, json, gzip, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KAGJAX = os.path.dirname(HERE)
R = os.path.dirname(KAGJAX)
if KAGJAX not in sys.path:
    sys.path.insert(0, KAGJAX)

LEADERS_DEFAULT = ("Boey|DSM|M & M & P & Q|Fourth Quadrant|DECEM|Unknown Mother-Goose|Azat Akhtyamov|seek inspiration|"
                   "Majkel1337|THIRD FARM CLUB|Vadim Vasilenko|吃白饭的大肥鱼|mtmr_s1|Kaggledew Valley 🏆|nah id win")
# players who beat our live agents (operator, Sep 27): their games among the leader tapes are TOP opponents too; our own
# games against them (and against everyone rated 2,400+) come from the specialist dirs (SPEC_DIRS, every non-"us:" seat)
SPECIALISTS_DEFAULT = "elmo|Aaweg|Yannik Schiffner|SpaTaro|lingxiaojun|yarneo|Snorlax|Russell Kirk"
TOP_TEAMS_DEFAULT = LEADERS_DEFAULT + "|" + SPECIALISTS_DEFAULT


def is_us(team):
    """Our own seat in the tapes of our live games (index teams "us:<name>"): never a tape opponent."""
    return str(team).startswith("us:")


def benchmark_episodes():
    """Every episode id mentioned by a benchmark / held-out file under il/live (never train on them)."""
    eps = set()
    for fn in glob.glob(f"{R}/il/live/*.json"):
        name = os.path.basename(fn)
        if not any(k in name for k in ("heldout", "top_worlds", "ladder")):
            continue
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        items = d if isinstance(d, list) else list(d.values()) if isinstance(d, dict) else []
        for x in items:
            if isinstance(x, int):
                eps.add(x)
            elif isinstance(x, dict) and "ep" in x:
                eps.add(int(x["ep"]))
            elif isinstance(x, (list, tuple)) and x and isinstance(x[0], str) and x[0].endswith(".json.gz"):
                try:
                    eps.add(int(os.path.basename(x[0]).split(".")[0]))
                except ValueError:
                    pass
    return eps


def select(dirs, leaders, split, max_n=None, any_opponent=False):
    """[(path, leader seat)] for tapes whose seat is a leader (team in leaders, "|"-separated; "*" = any team; our own
    "us:" seats never); any_opponent=True: every non-"us:" seat (the specialist dirs); split = "train" (ep % 10 != 0)
    or "heldout"."""
    bench = benchmark_episodes()
    lead = set(leaders.split("|"))
    out, seen = [], set()
    for d in dirs:
        idx = f"{d}/index.json"
        if not os.path.exists(idx):
            continue
        for ep, m in sorted(json.load(open(idx)).items()):
            e = int(ep)
            if e in bench or e in seen or ((e % 10 == 0) != (split == "heldout")):     # (an episode in two dirs: once)
                continue
            p = f"{d}/{ep}.json.gz"
            teams = m.get("teams") or []
            if len(teams) != 2 or not os.path.exists(p) or m.get("statuses", ["DONE", "DONE"]) != ["DONE", "DONE"]:
                continue
            seen.add(e)
            for s in (0, 1):
                if not is_us(teams[s]) and (any_opponent or "*" in lead or teams[s] in lead):
                    out.append((p, s))
    if max_n is not None and len(out) > max_n:
        rng = np.random.RandomState(0)
        out = [out[i] for i in sorted(rng.choice(len(out), max_n, replace=False))]
    return out, len(bench)


def build_bank(dirs=None, leaders=None, split="train", max_n=512, both=False, spec_dirs=(), spec_max=None):
    """both=True also stores the learner seat's recorded actions (l* arrays; test_tapes.py replays whole games).
    Leader seats of `dirs` (max_n of them), then the specialist seats of `spec_dirs` (every non-"us:" seat, spec_max of
    them); bank["spec"] bool [T] marks the latter (kind SPEC)."""
    import rng as RNG
    import state as AS
    dirs = [f"{R}/data/tapes_top12"] if dirs is None else list(dirs)
    leaders = leaders or os.environ.get("TOP_TEAMS", LEADERS_DEFAULT)
    sel, nbench = select(dirs, leaders, split, max_n) if dirs and max_n != 0 else ([], 0)
    nlead = len(sel)
    if spec_dirs:
        sp, nbench = select(list(spec_dirs), leaders, split, spec_max, any_opponent=True)
        sel = sel + [x for x in sp if x not in set(sel)]
    bank = bank_from_selection(sel, [i >= nlead for i in range(len(sel))], both)
    print(f"[tapes] {split}: {len(bank['lseat'])} tape seats = {len(bank['lseat']) - int(bank['spec'].sum())} leaders from {dirs} + "
          f"{int(bank['spec'].sum())} specialists from {list(spec_dirs)} ({bank['n_not_exact']} not reproducing exactly skipped; "
          f"benchmark episodes excluded: {nbench})", flush=True)
    return bank


DEVICE_KEYS = ("words", "uop", "uarg", "un", "mtype", "mitem", "mn", "lseat", "shops_rec", "e0_rec",
               "luop", "luarg", "lun", "lmtype", "lmitem", "lmn")


def device_arrays(bank, pinned=True):
    """The arrays step_tapes reads (host numpy); pinned=False drops the pinning arrays (natural words only)."""
    return {k: v for k, v in bank.items() if k in DEVICE_KEYS and (pinned or k not in ("shops_rec", "e0_rec"))}


def bank_from_selection(sel, spec_flags=None, both=False):
    """Device-bank arrays for an explicit selection [(path, leader seat)] (spec_flags [len(sel)] bool); recordings that do
    not reproduce exactly in the C++ engine are skipped (bank["n_not_exact"])."""
    import rng as RNG
    import state as AS
    spec_flags = [False] * len(sel) if spec_flags is None else list(spec_flags)
    sys.path.insert(0, f"{R}/tools")
    import topruler                                           # C++ both-seats replay: recorded shops + empty counts
    worlds, keep, spec = {}, [], []
    for i, (p, s) in enumerate(sel):
        if p not in worlds:
            d0 = json.load(gzip.open(p, "rt"))
            worlds[p] = topruler.recorded_world(p, d0)
        if worlds[p][2]:                                      # the recording reproduces exactly (else skipped)
            keep.append((p, s)); spec.append(bool(spec_flags[i]))
    nskip = len(sel) - len(keep)
    sel = keep
    T = len(sel)
    bank = dict(words=np.zeros((T, 30, 624), np.uint32), uop=np.zeros((T, 720, 40), np.int16), uarg=np.zeros((T, 720, 40), np.int16),
                un=np.ones((T, 720, 40), np.int16), mtype=np.zeros((T, 720, 10), np.int16), mitem=np.zeros((T, 720, 10), np.int16),
                mn=np.zeros((T, 720, 10), np.int32), lseat=np.zeros(T, np.int32),
                shops_rec=-np.ones((T, 8), np.int32), e0_rec=np.zeros((T, 30), np.int32), spec=np.asarray(spec, bool).reshape(T))
    if both:
        for k in ("uop", "uarg", "un", "mtype", "mitem", "mn"):
            bank["l" + k] = np.zeros_like(bank[k]) + (1 if k == "un" else 0)
    meta = []
    for i, (p, s) in enumerate(sel):
        d = json.load(gzip.open(p, "rt"))
        bank["words"][i] = RNG.game_words(int(d["seed"]), 30)
        for pre, seat in (("", s),) + ((("l", 1 - s),) if both else ()):
            acts = d["actions"][seat]
            for t in range(min(720, len(acts))):
                a = acts[t] if isinstance(acts[t], dict) else {"farmer": ["PASS"], "hands": [], "market": []}
                uop, uarg, un, mt, mi, mn = AS.encode_action(a)
                bank[pre + "uop"][i, t], bank[pre + "uarg"][i, t], bank[pre + "un"][i, t] = uop, uarg, np.clip(un, -30000, 30000)
                bank[pre + "mtype"][i, t], bank[pre + "mitem"][i, t], bank[pre + "mn"][i, t] = mt, mi, mn
        bank["lseat"][i] = 1 - s
        shops, e0, _ = worlds[p]
        bank["shops_rec"][i, :len(shops[:8])] = list(shops)[:8]
        bank["e0_rec"][i, :len(e0[:30])] = list(e0)[:30]
        rw = d.get("rewards") or [None, None]
        meta.append((int(d["ep"]), (d.get("teams") or ["?", "?"])[s], s, rw[s], rw[1 - s], bool(spec[i])))
    bank["meta"] = meta
    bank["paths"] = [p for p, _ in sel]
    bank["n_not_exact"] = nskip
    return bank


def check(argv):
    """python tapes.py check [--dirs d1,d2] [--spec d3] [--teams ...]: what the trainer would select from these dirs
    (train / held-out seats, teams, benchmark exclusions) and whether every selected recording reproduces exactly in the
    C++ engine (topruler.recorded_world), i.e. whether the tape worlds can be pinned."""
    import argparse, collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=f"{R}/data/tapes_top12")
    ap.add_argument("--spec", default="")
    ap.add_argument("--teams", default=TOP_TEAMS_DEFAULT)
    ap.add_argument("--replay", type=int, default=1)
    a = ap.parse_args(argv)
    sys.path.insert(0, f"{R}/tools")
    import topruler
    out = {}
    for label, dirs, anyo in (("leaders", [d for d in a.dirs.split(",") if d], False), ("specialists", [d for d in a.spec.split(",") if d], True)):
        if not dirs:
            continue
        for split in ("train", "heldout"):
            sel, nb = select(dirs, a.teams, split, None, any_opponent=anyo)
            teams = collections.Counter()
            ok = bad = 0
            for p, s_ in sel:
                d = json.load(gzip.open(p, "rt"))
                teams[(d.get("teams") or ["?", "?"])[s_]] += 1
                if a.replay:
                    ok_ = topruler.recorded_world(p, d)[2]
                    ok += bool(ok_); bad += (not ok_)
            out[f"{label}/{split}"] = dict(seats=len(sel), episodes=len({p for p, _ in sel}), distinct_teams=len(teams),
                                           reproduce_exactly=ok, not_exact=bad, top_teams=teams.most_common(8))
            print(f"[{label} {split}] {dirs}: {len(sel)} seats in {len({p for p, _ in sel})} episodes, {len(teams)} teams "
                  f"(top: {teams.most_common(6)}); C++ replay exact {ok}, not exact {bad}; benchmark episodes excluded: {nb}",
                  flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(sys.argv[2:])
