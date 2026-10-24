"""Server (operator, Sep 25): one Kaggle daily top-games zip (kaggle/kaggriculture-episodes-<day>) -> compact tapes,
streamed from the zip (never extracted), W worker processes each reading its own share of the members.
Games already taped in <out_dir> or data/tapes_top12 are skipped.  Updates <out_dir>/index.json.
Usage: srv_import_day.py <zip> <day> <out_dir> [W]"""
import sys, os, json, gzip, zipfile, time
import multiprocessing as mp
R = "/Users/asad.ali/kaggriculture"


def convert(eid, d, day):                                 # = pull_leaders.convert
    steps = d["steps"]
    acts = [[(steps[t + 1][s].get("action") if t + 1 < len(steps) and isinstance(steps[t + 1][s].get("action"), dict) else None)
             for t in range(720)] for s in (0, 1)]
    return dict(ep=eid, seed=int(d["info"]["seed"]), teams=d["info"].get("TeamNames"), rewards=d["rewards"],
                statuses=[steps[-1][s].get("status") for s in (0, 1)],
                shops=list(steps[-1][0]["observation"]["town"]["unlocked_shops"]), actions=acts, day=day)


def work(args):
    import orjson
    zp, names, day, out = args
    z = zipfile.ZipFile(zp); idx = {}; bad = 0
    for name in names:
        eid = int(os.path.basename(name)[:-5])
        try:
            d = orjson.loads(z.read(name)); t = convert(eid, d, day); del d
        except Exception as ex:
            bad += 1; continue
        with gzip.open(f"{out}/{eid}.json.gz", "wt") as fh:
            json.dump(t, fh, separators=(",", ":"))
        idx[str(eid)] = dict(teams=t["teams"], rewards=t["rewards"], day=day, statuses=t["statuses"])
    return idx, bad


if __name__ == "__main__":
    zp, day, out = sys.argv[1], sys.argv[2], sys.argv[3]; W = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.makedirs(out, exist_ok=True); t0 = time.time()
    known = set()
    import glob
    for d_ in {out, f"{R}/data/tapes_top12", *glob.glob(f"{R}/data/tapes_kd/*")}:
        fn = f"{d_}/index.json"
        if os.path.exists(fn):
            known |= {e for e in json.load(open(fn)) if os.path.exists(f"{d_}/{e}.json.gz")}
    names = [n for n in zipfile.ZipFile(zp).namelist() if n.endswith(".json") and os.path.basename(n)[:-5].isdigit()]
    todo = [n for n in names if os.path.basename(n)[:-5] not in known]
    print(f"{day}: {len(names)} games in the zip, {len(todo)} new", flush=True)
    with mp.get_context("spawn").Pool(W) as pool:
        res = pool.map(work, [(zp, todo[k::W], day, out) for k in range(W)])
    fn = f"{out}/index.json"; idx = json.load(open(fn)) if os.path.exists(fn) else {}
    for part, _ in res:
        idx.update(part)
    json.dump(idx, open(fn, "w"), ensure_ascii=False)
    print(f"{day}: {sum(len(p) for p, _ in res)} taped, {sum(b for _, b in res)} failed, index {len(idx)}, {time.time()-t0:.0f}s", flush=True)
