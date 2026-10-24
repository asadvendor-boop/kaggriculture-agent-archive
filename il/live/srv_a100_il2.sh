#!/bin/bash
# Sep 27 night, A100 box (operator: "weighted IL ... with the top leaders' current replays"): once il/data/fresh3 is ready
# (srv_fresh3.sh: fresh2 + the leaders' newest games, rating weights), two rating-weighted imitation runs on the GPU
# (it shares the GPU with the anti-specialist RL run), each followed by the DSM fine-tune (2 epochs, as v13_p3) and the
# rulers with the SE guard (srv_a100_eval.sh, paired vs v13_p3):
#   n1 = v13_p3's pass-stage model (ilsp_v13_p3_p) + 1 pass over fresh3 at 1e-4          (~40 min)
#   n3 = v13_p3's exact recipe on the new data: g20_w5_s0 + 3 passes over fresh3 at 1.5e-4 (~2 h)
# N_TEAMS=50: the team vocabulary gained the 2 teams new in the top 10 (srv_fresh3.sh); 48-team inits are expanded.
# Progress: /root/a100_il.log.
# srv_a100_il2 (Sep 27 ~22:20 PKT, operator: "counter the remaining weakness ... tonight too"): as srv_a100_il.sh, plus the
# leaders' WINS against the specialists who beat our live 20M (every opponent that beat us in data/tapes_spec +
# tapes.SPECIALISTS_DEFAULT + data/tapes_spec_more/teams.txt when present) get DOUBLE rating weight in fresh3.
exec 8>/tmp/srv_a100_il.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; LOG=/root/a100_il.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="BLOCKS=10 N_TEAMS=50 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1"
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
until grep -q "FRESH3 READY" /root/fresh3.log 2>/dev/null; do sleep 60; done
say "fresh3 ready: $(ls data/fresh3 | wc -l) seats"
$PY - >> $LOG 2>&1 <<'PY'
import json, os, sys
K = "/Users/asad.ali/kaggriculture"; sys.path.insert(0, f"{K}/kagjax/train")
spec = set()
for v in json.load(open(f"{K}/data/tapes_spec/index.json")).values():
    t = v["teams"]; o = t[1] if str(t[0]).startswith("us:") else t[0]
    if v.get("res_vs_us") == "L": spec.add(o)
try:
    import tapes; spec |= set(tapes.SPECIALISTS_DEFAULT.split("|"))
except Exception: pass
if os.path.exists(f"{K}/data/tapes_spec_more/teams.txt"): spec |= set(open(f"{K}/data/tapes_spec_more/teams.txt").read().split("|"))
idx = {}
for d in ("data/tapes_top12", "data/tapes_kd/2026-09-25"):
    try: idx.update(json.load(open(f"{K}/{d}/index.json")))
    except Exception: pass
p = f"{K}/il/data/rating_w_fresh3.json"; w = json.load(open(p)); n = 0
for key in list(w):
    ep, s = key.rsplit("_", 1); m = idx.get(ep)
    if not m or not m.get("teams") or not m.get("rewards") or None in m["rewards"]: continue
    s = int(s); opp = m["teams"][1 - s]
    if opp in spec and m["rewards"][s] > m["rewards"][1 - s]:
        w[key] *= 2.0; n += 1
json.dump(w, open(p, "w"))
print(f"specialist boost: {len(spec)} specialist teams; {n} leader seats that beat one of them now weigh x2", flush=True)
PY
run() {   # run <name> <init> <passes> <lr>
  local n=$1 log=logs/ilsp_$1.log
  say "$n: $3 pass(es) of fresh3 from $2 lr $4"
  env SEED=0 MAX_LR=$4 RATING_W=$K/il/data/rating_w_fresh3.json INIT=$2 $C $PY train6.py ckpt/ilsp_${n}_p.pt $3 320 data/fresh3 >> $log 2>&1
  say "$n pass stage: $(grep 'best val loss' $log | tail -1)"
  env SEED=0 MAX_LR=2e-4 INIT=ckpt/ilsp_${n}_p.pt $C $PY train6.py ckpt/ilsp_${n}.pt 2 320 data/dsm_ft2 >> $log 2>&1
  say "$n DONE: $(grep 'best val loss' $log | tail -1)"
  setsid -f bash live/srv_a100_eval.sh ilsp_$n > /dev/null 2>&1 < /dev/null
}
run n1 ckpt/ilsp_v13_p3_p.pt 1 1e-4
run n3 ckpt/g20_w5_s0.pt 3 1.5e-4
say "IL runs done"
