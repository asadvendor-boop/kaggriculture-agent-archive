import gzip, json, glob, os, sys, collections
D = "/Users/asad.ali/kaggriculture/data"
# ep -> (end, rating, opp_rating, sub) from rows files
end = {}
for f in glob.glob(f"{D}/rows_*.json"):
    try:
        for r in json.load(open(f)):
            end[int(r["ep"])] = (r.get("end"), r.get("rating"), r.get("opp_rating"), os.path.basename(f)[5:-5])
    except Exception as e:
        pass
print("rows eps", len(end))
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{D}/{d}/*.json.gz"))
    c = collections.Counter(); names = collections.Counter(); dates = []; opp_r = []; margins = []; subs = collections.Counter()
    for f in fs:
        try:
            x = json.load(gzip.open(f, "rt"))
        except Exception:
            c["bad"] += 1; continue
        s = x.get("seat"); nm = x.get("names") or [None, None]
        ours = nm[s] if s in (0, 1) else None
        names[ours] += 1
        rw = x.get("rewards") or [None, None]
        if s not in (0, 1) or rw[0] is None or rw[1] is None:
            c["no_reward"] += 1; continue
        m = rw[s] - rw[1 - s]
        c["W" if m > 0 else ("L" if m < 0 else "T")] += 1
        if x.get("res") and x["res"] != ("W" if m > 0 else "L" if m < 0 else "T"):
            c["res_mismatch"] += 1
        margins.append(m)
        e = end.get(int(x["ep"]))
        if e:
            dates.append(e[0]); subs[e[3]] += 1
        if x.get("opp_rating") is not None: opp_r.append(x["opp_rating"])
    dates = sorted(d_ for d_ in dates if d_)
    import statistics as st
    print(f"{d}: n {len(fs)} {dict(c)} ours={dict(names.most_common(3))} dates {dates[0] if dates else '-'}..{dates[-1] if dates else '-'} (with date {len(dates)}) subs {dict(subs)} opp_rating med {st.median(opp_r) if opp_r else '-'} min {min(opp_r) if opp_r else '-'} |margin|<1k {sum(abs(m)<1000 for m in margins)}")
