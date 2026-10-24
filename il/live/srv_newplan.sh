#!/bin/bash
# Sep 28 ~03:25 PKT: learn the leaders' NEW build order natively (PLANNING_DESIGN.md 2.2: on Sep 26 DSM switched to M&M&P&Q's
# plan - no third land, SW on day ~8.2, one cow a day on days 2-4 with the pasture layout for it, a melon wave on NE on day
# 6). The market-rule versions can't place early cows (our policy's layout has no pasture before day 6), so imitate the
# plan itself: il/data/newplan = every DSM and M & M & P & Q seat of Sep 26-27 (~1,100, benchmark games already excluded
# from fresh3) + 2,000 random other fresh3 seats (teams < 48, style anchor) -> v13_p3 + 1 gentle epoch at 1e-4 (after n3
# has released the GPU) -> ckpt/ilsp_v13_newplan.pt -> quick behaviour check (live/rule_smoke.py, no rules: land days,
# cows by day, late melons) -> srv_a100_eval.sh (leader tests, live losses, 464; paired vs v13_p3). Log /root/newplan.log
exec 8>/tmp/srv_newplan.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; LOG=/root/newplan.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
$PY - >> $LOG 2>&1 <<'PY'
import os, glob, json, random
import numpy as np
K = "/Users/asad.ali/kaggriculture"; R = f"{K}/il/data"; out = f"{R}/newplan"; os.makedirs(out, exist_ok=True)
voc = json.load(open(f"{R}/team_vocab.json"))
day = {}
for f in glob.glob(f"{K}/data/tapes_*/index.json") + glob.glob(f"{K}/data/tapes_kd/*/index.json"):
    try:
        for ep, v in json.load(open(f)).items():
            if v.get("day"): day[int(ep)] = v["day"]
    except Exception:
        pass
plan, other = [], []
for f in sorted(glob.glob(f"{R}/fresh3/*.npz")):
    b = os.path.basename(f); ep = int(b.split("_")[0])
    if not os.path.exists(f"{R}/x6/{b[:-4]}.npy"): continue
    ti = int(np.load(f)["meta"][6])
    team = voc[ti] if ti < len(voc) else "?"
    if team in ("DSM", "M & M & P & Q") and day.get(ep, "") >= "2026-09-26":
        plan.append(f)
    elif ti < 48:
        other.append(f)
random.Random(28).shuffle(other); anchor = other[:2000]
for f in plan + anchor:
    b = os.path.basename(f)
    if not os.path.lexists(f"{out}/{b}"): os.symlink(os.path.realpath(f), f"{out}/{b}")
print(f"newplan: {len(plan)} DSM/MMPQ seats of Sep 26-27 + {len(anchor)} anchor seats -> {len(os.listdir(out))} seats "
      f"(days known for {len(day)} episodes)", flush=True)
PY
say "waiting for the GPU (n3 must finish first)"
while pgrep -f "python train6.py" > /dev/null; do sleep 60; done
say "fine-tune: v13_p3 + 1 epoch of newplan at 1e-4"
cd il
env SEED=0 MAX_LR=1e-4 INIT=ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 \
  $PY train6.py ckpt/ilsp_v13_newplan.pt 1 320 data/newplan > logs/ilsp_v13_newplan.log 2>&1
say "NEWPLAN DONE: $(grep 'best val loss' logs/ilsp_v13_newplan.log | tail -1)"
for m in ilsp_v13_p3 ilsp_v13_newplan; do
  RULES_ENV="SE_GUARD=1" nice -n 5 $PY live/rule_smoke.py ckpt/$m.pt 4 fresh2 2>&1 | grep -E "^== ep|d0[2-9]:|d1[0-1]:" | sed "s/^/  [$m] /" >> $LOG
done
setsid -f bash live/srv_a100_eval.sh ilsp_v13_newplan > /dev/null 2>&1 < /dev/null 8>&-
