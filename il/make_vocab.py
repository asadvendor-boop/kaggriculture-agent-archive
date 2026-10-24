# Team-style vocabulary: the 11 current leaders keep ids 0-10 (so v3d's style embeddings carry over); every other team
# with >= MIN seats in the top-rated daily games + older top sets is appended by frequency.
import json, collections, sys
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
LEAD = ["Boey", "M & M & P & Q", "nah id win", "DECEM", "Unknown Mother-Goose", "Vadim Vasilenko", "吃白饭的大肥鱼",
        "THIRD FARM CLUB", "Kaggledew Valley 🏆", "mtmr_s1", "DSM"]
c = collections.Counter()
for d_ in ("data/tapes_daily", "data/tapes_0922", "data/tapes_archive"):
    for ep, m in json.load(open(f"/Users/asad.ali/kaggriculture/{d_}/index.json")).items():
        for t in (m.get("teams") or []):
            if t: c[t] += 1
vocab = LEAD + [t for t, n in c.most_common() if n >= MIN and t not in LEAD]
json.dump(vocab, open("/Users/asad.ali/kaggriculture/il/data/team_vocab.json", "w"), ensure_ascii=False, indent=0)
print(len(vocab), "teams:", ", ".join(f"{i}:{t[:12]}" for i, t in enumerate(vocab)))
