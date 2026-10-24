#!/bin/bash
# Sep 27 night (operator: "counter the remaining weakness in the current build ... tonight too"): v13_p3 (NEURAL_20MW8_SE)
# turned 12 of the live 20M's 28 recorded specialist losses into wins, but SpaTaro, Aaweg, sekai013 & co still beat it.
#  1. refresh our live submissions' game lists (public ListEpisodes) -> who beat us most recently
#  2. ~60 recent games of every specialist submission rated >= 2,200 that beat us (public endpoints, 8 threads, backoff)
#     -> data/tapes_spec_more (+ teams.txt)          (after srv_fresh3.sh's own game lists are done: no parallel listing)
#  3. the anti-specialist RL restarts from its newest state with those games (the specialist plays its recorded moves, our
#     agent the other seat) and the top 10's newest games as tape opponents; progress /root/spec_more.log
exec 8>/tmp/srv_spec_more_r2.lock; flock -n 8 || { echo "already running"; exit 0; }   # r2: the RL it restarted inherited the first lock
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; JPY=/opt/jaxenv/bin/python; LOG=/root/spec_more.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
until grep -q "round 2" /root/fresh3.log 2>/dev/null; do sleep 30; done
# (Sep 27 ~22:35 PKT re-run: the first run fetched 0 games - fetch_specialist_games.py accepted only SUCCEEDED episodes,
#  the public listing says COMPLETED; fixed. RL now resumes from the NEWEST state of any antispec* run.)
$PY - >> $LOG 2>&1 <<'PY'
import sys, json; sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); import pull
for s in (56595847, 56589891):
    try:
        L = pull.fetch(s); R = pull.rows_of(L, s); json.dump(R, open(f"/Users/asad.ali/kaggriculture/data/rows_{s}.json", "w"))
        print(f"our submission {s}: {len(R)} games, {sum(r.get('res') == 'L' for r in R)} losses", flush=True)
    except Exception as e:
        print(f"our submission {s}: listing failed {str(e)[:100]}", flush=True)
PY
$PY il/live/fetch_specialist_games.py --out data/tapes_spec_more --per 60 --minr 2200 --threads 8 >> $LOG 2>&1
say "specialist games: $($PY -c "import json; print(len(json.load(open('data/tapes_spec_more/index.json'))))") tapes; teams: $(tr '|' ',' < data/tapes_spec_more/teams.txt | cut -c1-400)"
# restart the RL from its newest state with the new tape opponents
ST=$(ls -t kagjax/runs/antispec*/state_*.npz 2>/dev/null | head -1); NEXT=antispec$(( $(ls -d kagjax/runs/antispec* | wc -l) + 1 ))
P=$(ps -eo pid,args | awk '$2 ~ /python/ && $3 ~ /kagjax\/train\/ppo.py$/ {print $1}' | head -1)
[ -z "$ST" ] && { say "no RL state yet - RL not restarted"; exit 0; }
LEAD=$($JPY -c "import sys; sys.path.insert(0, '$K/kagjax/train'); import tapes; print(tapes.TOP_TEAMS_DEFAULT)")
TT="$LEAD|Just A game on your lips|Anton Tikhonov|$(cat data/tapes_spec_more/teams.txt)"
say "RL restart: stopping PID $P, resuming from $ST"
[ -n "$P" ] && kill $P; sleep 20
setsid -f bash kagjax/train/run_gpu.sh $NEXT PRESET=gate RESUME=$K/$ST INIT=$K/il/ckpt/ilsp_v13_p3.pt IL=$K/il/ckpt/ilsp_v13_p3_p.pt \
  TEAM=10 T=0.2 KSHARP=10 TAU=0.5 BETA=0.05 WINB=0.3 ENT_COEF=0 OPPS=SPEC=4,TOP=5,IL=1 \
  TAPE_DIRS=$K/data/tapes_top12,$K/data/tapes_spec_more SPEC_DIRS=$K/data/tapes_spec TOP_TEAMS="$TT" TAPE_MAX=4096 SPEC_MAX=1024 \
  B=1024 K=4 SEG=720 SE_GUARD=1 EVAL_EVERY=3 EVAL_N=128 EVAL_OPPS=SNAP0,IL@10 PT_EVERY=2 CKPT_EVERY=2 ITERS=1000 SEED=301 \
  > /root/${NEXT}_launch.log 2>&1 < /dev/null 8>&-          # never hand our lock to the long-running RL
sleep 5; say "RL restarted: $(tail -1 /root/${NEXT}_launch.log)"
