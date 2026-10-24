#!/bin/bash
# Sep 27 night (operator: "keep building counters to our both current scripts' weakness as they arrive on the live
# ladder"): every 2 hours, if the specialist bank (data/tapes_spec, where srv_poll_live.sh appends every new live loss of
# both active submissions) has grown since the last RL (re)start, the anti-specialist RL restarts from its NEWEST state
# so its opponents include the new losses (the tape bank is built at start-up). Same settings as srv_spec_more.sh.
# Log: /root/rl_refresh.log
exec 9>/tmp/srv_rl_refresh.lock; flock -n 9 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K; PY=/venv/main/bin/python; JPY=/opt/jaxenv/bin/python; LOG=/root/rl_refresh.log
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
nbank() { $PY -c "import json; print(len(json.load(open('$K/data/tapes_spec/index.json'))))"; }
LAST=$(nbank); say "armed: specialist bank $LAST games; first check in 2 h"
while true; do
  sleep 7200
  N=$(nbank)
  if [ "$N" -le "$LAST" ]; then say "bank $N games: no new losses, RL left running"; continue; fi
  ST=$(ls -t kagjax/runs/antispec*/state_*.npz 2>/dev/null | head -1); NEXT=antispec$(( $(ls -d kagjax/runs/antispec* | wc -l) + 1 ))
  P=$(ps -eo pid,args | awk '$2 ~ /python/ && $3 ~ /kagjax\/train\/ppo.py$/ {print $1}' | head -1)
  [ -z "$ST" ] && { say "no RL state - skipped"; continue; }
  LEAD=$($JPY -c "import sys; sys.path.insert(0, '$K/kagjax/train'); import tapes; print(tapes.TOP_TEAMS_DEFAULT)")
  TT="$LEAD|Just A game on your lips|Anton Tikhonov|$(cat data/tapes_spec_more/teams.txt 2>/dev/null)"
  say "bank $LAST -> $N games: restarting the RL (PID $P) from $ST as $NEXT"
  [ -n "$P" ] && kill $P; sleep 20
  setsid -f bash kagjax/train/run_gpu.sh $NEXT PRESET=gate RESUME=$K/$ST INIT=$K/il/ckpt/ilsp_v13_p3.pt IL=$K/il/ckpt/ilsp_v13_p3_p.pt \
    TEAM=10 T=0.2 KSHARP=10 TAU=0.5 BETA=0.05 WINB=0.3 ENT_COEF=0 OPPS=SPEC=4,TOP=5,IL=1 \
    TAPE_DIRS=$K/data/tapes_top12,$K/data/tapes_spec_more SPEC_DIRS=$K/data/tapes_spec TOP_TEAMS="$TT" TAPE_MAX=4096 SPEC_MAX=1024 \
    B=1024 K=4 SEG=720 SE_GUARD=1 EVAL_EVERY=3 EVAL_N=128 EVAL_OPPS=SNAP0,IL@10 PT_EVERY=2 CKPT_EVERY=2 ITERS=1000 SEED=301 \
    > /root/${NEXT}_launch.log 2>&1 < /dev/null 9>&-
  sleep 5; say "RL restarted: $(tail -1 /root/${NEXT}_launch.log)"; LAST=$N
done
