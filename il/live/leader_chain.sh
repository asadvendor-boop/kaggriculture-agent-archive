#!/bin/bash
# After the top-team downloads: index repair -> seats -> v6 extras -> per-leader v6 fine-tunes -> full-ladder benchmarks.
cd /Users/asad.ali/kaggriculture
until grep -q "fetch done" il/logs/fetch_top13.log && grep -q "fetch done" il/logs/fetch_top13b.log; do sleep 30; done
echo "[$(date +%H:%M)] downloads done"
.venv/bin/python il/live/rebuild_pull_index.py
for w in 0 1 2 3; do nice -n 5 .venv/bin/python il/build_dataset3.py il/data/all $w 4 data/tapes_pull > il/logs/build_top13_w$w.log 2>&1 & done; wait
echo "[$(date +%H:%M)] seats: $(tail -q -n1 il/logs/build_top13_w*.log | awk '{s+=$6} END {print s}') written"
for w in 0 1 2 3; do nice -n 5 .venv/bin/python il/build_extras6.py $w 4 > il/logs/x6_top13_w$w.log 2>&1 & done; wait
echo "[$(date +%H:%M)] extras done: $(ls il/data/x6 | wc -l) seats"
.venv/bin/python - <<'PY'
import glob, os, numpy as np, collections
R = "/Users/asad.ali/kaggriculture"; X6 = f"{R}/il/data/x6"
want = {10: "dsm", 1: "mmpq", 3: "decem", 5: "vadim"}; n = collections.Counter()
for tid, name in want.items():
    os.makedirs(f"{R}/il/data/lead_{name}", exist_ok=True)
for d in ("leaders", "all"):
    for f in glob.glob(f"{R}/il/data/{d}/*.npz"):
        try:
            m = np.load(f)["meta"]
        except Exception:
            continue
        tid = int(m[6]); b = os.path.basename(f)
        if tid in want and os.path.exists(f"{X6}/{b[:-4]}.npy"):
            dst = f"{R}/il/data/lead_{want[tid]}/{b}"
            if not os.path.exists(dst): os.symlink(f, dst)
            n[want[tid]] += 1
print("leader seats:", dict(n))
PY
cd il
for L in dsm mmpq decem; do
  N_TEAMS=47 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 INIT=ckpt/bc_v6.pt nice -n 3 ../.venv/bin/python train6.py ckpt/bc_v6_$L.pt 2 96 data/lead_$L > logs/train_v6_$L.log 2>&1
  echo "[$(date +%H:%M)] fine-tune $L: $(grep 'best val loss' logs/train_v6_$L.log)"
done
cd ..
for pair in "dsm 10" "mmpq 1" "decem 3"; do set -- $pair
  LOCKSTEP=6 NW=8 TEAM=$2 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3 nice -n 3 .venv/bin/python il/live/ladder.py eval test il/ckpt/bc_v6_$1.pt 2>&1 | grep -v Warning | grep -v recorded
done
echo "[$(date +%H:%M)] leader chain done"
