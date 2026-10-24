#!/bin/bash
# Sep 28 ~02:15 PKT (operator: "do all trainings, experiments, improvements and gap filling tonight"): NEURAL_20MW8_SE's
# first 8 live losses were to 2,700+ players who out-produce us on melon / milk / wool (net cash per game: melon -3.1k,
# milk -2.0k, wool -1.7k) - many small planting/volume choices, not one exact rule. So: learn from the players who beat
# us. Their 3,584 recent games (data/tapes_spec_more, srv_spec_more.sh) -> training seats (all 54 teams added to the
# vocabulary) -> il/data/specft = their WINNING seats (benchmark games excluded) + DSM's fine-tune seats (style anchor)
# -> a short gentle fine-tune of v13_p3 (NEURAL_20MW8_SE): 1 epoch at 1e-4 -> ckpt/ilsp_v13_specft.pt -> srv_a100_eval.sh
# (leader tests, live-loss replay, 464; paired vs v13_p3). Log: /root/specft.log
exec 8>/tmp/srv_specft.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; LOG=/root/specft.log; NW=16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
NT=$($PY - <<'PY'
import json
p = "/Users/asad.ali/kaggriculture/il/data/team_vocab.json"; v = json.load(open(p))
for t in open("/Users/asad.ali/kaggriculture/data/tapes_spec_more/teams.txt").read().split("|"):
    if t and t not in v: v.append(t)
json.dump(v, open(p, "w"), ensure_ascii=False); print(len(v))
PY
)
say "team vocabulary: $NT teams"
mkdir -p il/data/spec_all
for w in $(seq 0 $((NW - 1))); do nice -n 5 $PY il/build_dataset3.py il/data/spec_all $w $NW data/tapes_spec_more > il/logs/build_spec_w$w.log 2>&1 & done; wait
say "seats: $(tail -q -n1 il/logs/build_spec_w*.log | awk '{s+=$8} END {print s}') written, non-exact/failed $(tail -q -n1 il/logs/build_spec_w*.log | awk '{s+=$10} END {print s}')"
for w in $(seq 0 $((NW - 1))); do TAPES=data/tapes_spec_more SEATS=il/data/spec_all nice -n 5 $PY il/live/srv_build_x6.py $w $NW > il/logs/x6_spec_w$w.log 2>&1 & done; wait
say "v6 features: $(tail -q -n1 il/logs/x6_spec_w*.log | awk '{s+=$5} END {print s}') new"
$PY - >> $LOG 2>&1 <<'PY'
import os, glob, json
import numpy as np
K = "/Users/asad.ali/kaggriculture"; R = f"{K}/il/data"; out = f"{R}/specft"; os.makedirs(out, exist_ok=True)
voc = json.load(open(f"{R}/team_vocab.json")); spec = set(t for t in open(f"{K}/data/tapes_spec_more/teams.txt").read().split("|") if t)
held = set(json.load(open(f"{K}/il/live/fresh_heldout_eps.json")))
held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/ladder_worlds.json"))}
for f in ("top_worlds_fresh_300.json", "top_worlds_fresh2_300.json", "top_worlds_livelosses_300.json"):
    try: held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/{f}"))}
    except Exception: pass
n = {"spec_wins": 0, "held_out": 0, "not_spec_or_lost": 0, "dsm": 0}
for f in glob.glob(f"{R}/spec_all/*.npz"):
    b = os.path.basename(f); ep = int(b.split("_")[0])
    if ep in held: n["held_out"] += 1; continue
    if not os.path.exists(f"{R}/x6/{b[:-4]}.npy"): continue
    m = np.load(f)["meta"]; team = voc[int(m[6])] if int(m[6]) < len(voc) else "?"
    if team in spec and int(m[7]) == 1:
        if not os.path.lexists(f"{out}/{b}"): os.symlink(f, f"{out}/{b}")
        n["spec_wins"] += 1
    else:
        n["not_spec_or_lost"] += 1
for f in glob.glob(f"{R}/dsm_ft2/*.npz"):
    b = os.path.basename(f)
    if not os.path.lexists(f"{out}/{b}"): os.symlink(os.path.realpath(f), f"{out}/{b}")
    n["dsm"] += 1
print(f"specft: {n} -> {len(os.listdir(out))} seats", flush=True)
PY
say "fine-tune: v13_p3 + 1 epoch of specft at 1e-4 (N_TEAMS=$NT)"
cd il
env SEED=0 MAX_LR=1e-4 INIT=ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=$NT MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 \
  $PY train6.py ckpt/ilsp_v13_specft.pt 1 320 data/specft > logs/ilsp_v13_specft.log 2>&1
say "SPECFT DONE: $(grep 'best val loss' logs/ilsp_v13_specft.log | tail -1)"
setsid -f bash live/srv_a100_eval.sh ilsp_v13_specft > /dev/null 2>&1 < /dev/null 8>&-
