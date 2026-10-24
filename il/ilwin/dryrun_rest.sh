#!/bin/bash
# Mac dry run, steps 2-5 (each gated by the 4-process cap): a few CPU training steps from ilsp_v13_p3 with rehearsal read in
# place from il/data (read-only), rollout play, agreement check, package export + Kaggle-style load.
S=$(cd "$(dirname "$0")" && pwd); PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; K=/Users/asad.ali/kaggriculture
D=$S/dryrun; cd $K
until grep -q "BUILD WALL" $D/build.log 2>/dev/null; do sleep 10; done
export OMP_NUM_THREADS=2 TORCH_THREADS=2
echo "== TRAIN"; bash $S/wait_slot.sh; T0=$(date +%s)
env INIT=$K/il/ckpt/ilsp_v13_p3.pt MAX_LR=5e-5 DEVICE=cpu BS=16 CHUNK=4 MAX_STEPS=6 SAVE_EVERY=0.008 VAL_MAX=2 VAL_STATES=256 \
    LOAD_THREADS=1 PREFETCH=0 BLOCKS=10 N_TEAMS=48 WIN_DIRS=$D/ds/seats X6_DIRS=$D/ds/x6:$K/il/data/x6 WIN_W=$D/ds/win_weights.json \
    REH="$K/il/data/fresh2:1.0,$K/il/data/dsm_ft2:0.5" WIN_TEAM_MODE=ours \
    nice -n 15 $PY $S/train_winner.py $D/ilsp_v13_win_dry.pt 2 320
echo "TRAIN WALL $(( $(date +%s) - T0 )) s"
echo "== PLAY"; bash $S/wait_slot.sh; T0=$(date +%s)
nice -n 15 $PY $S/dry_play.py $D/ilsp_v13_win_dry.pt $K/il/ckpt/ilsp_v13_p3.pt
echo "PLAY WALL $(( $(date +%s) - T0 )) s"
echo "== AGREE"; bash $S/wait_slot.sh; T0=$(date +%s)
nice -n 15 $PY $S/agree_check.py $D/ilsp_v13_win_dry.pt --set win=$D/ds/seats --set reh=$K/il/data/dsm_ft2 --x6 $D/ds/x6 \
    --all-files --max-files 4 --n 400 --device cpu --out $D/agree_dry.json
echo "AGREE WALL $(( $(date +%s) - T0 )) s"
echo "== EXPORT"; bash $S/wait_slot.sh; T0=$(date +%s); mkdir -p $D/pkg
(cd $K/il && B64=1 RULES="SE_GUARD=1,STRADDLE=2" RULES_STR="SELL_ORDER=held" nice -n 15 $PY export.py $D/ilsp_v13_win_dry.pt $D/pkg/main.py 0.5 10 2>&1 | grep -v -i warn | tail -5)
nice -n 15 $PY $S/dry_pkg.py $D/pkg/main.py
echo "EXPORT WALL $(( $(date +%s) - T0 )) s"
echo "DRYRUN REST DONE"
