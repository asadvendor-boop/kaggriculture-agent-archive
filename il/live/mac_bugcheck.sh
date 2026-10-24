#!/bin/bash
# Sep 29 bug check (A and S both collapse vs MMPQ / Boey on fresh4): (1) ZERO = v13_p3 through the trainer at LR ~0 (1 step) -> fresh4 must be
# EXACTLY 177-123 / +17,630; (2) CTRL = same pipeline and steps as S but on v13_p3's own DSM fine-tune data (dsm_ft2 as "winner" seats, team 10,
# rehearsal fresh2) -> fresh4 should stay ~177. Log il/logs/bugcheck29.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; LOG=$K/il/logs/bugcheck29.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG; }
until grep -qE "S CHAIN DONE|TRAIN S FAILED" logs/win29_chain.log; do sleep 20; done
export OMP_NUM_THREADS=1
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12"
TR="DEVICE=mps SEED=0 INIT=ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=50 VAL_MAX=180 VAL_STATES=20000 PROGRESS_EVERY=100 AUX_W=0.1 LOAD_THREADS=2 PREFETCH=1 WIN_TEAM_MODE=ours SAVE_EVERY=1"
f4() { c=$1; env ROLLOUT_DEVICE=mps LOCKSTEP=100 $ALL STRADDLE_PREBUY=1 DAY=fresh4 N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg \
        nice -n 10 $PY live/top_bench.py ckpt/$c.pt > logs/mac_fresh4_$c.log 2>&1
  $PY live/pair_cmp.py logs/top_fresh4_ilsp_v13_p3_t10_soh_st2_wg.json logs/top_fresh4_${c}_t10_soh_st2_wg.json 2>&1 | head -2 | tee -a $LOG
  tail -2 logs/mac_fresh4_$c.log | tail -1 | cut -c1-160 | tee -a $LOG; }
say "ZERO: v13_p3 through the trainer, 1 step at LR 1e-20"
env $TR MAX_LR=1e-20 MAX_STEPS=1 WIN_DIRS=data/win29s/seats X6_DIRS=data/win29s/x6:data/x6 REH="data/fresh2:1.0,data/dsm_ft2:0.5" \
    nice -n 10 $PY ilwin/train_winner.py ckpt/ilsp_v13_zero.pt 1 320 > logs/bugcheck_zero_train.log 2>&1 || say "ZERO train failed: $(tail -2 logs/bugcheck_zero_train.log)"
$PY - <<'PYEOF' 2>&1 | tee -a $LOG
import torch
a = torch.load("ckpt/ilsp_v13_p3.pt", map_location="cpu", weights_only=False)["state"]; b = torch.load("ckpt/ilsp_v13_zero.pt", map_location="cpu", weights_only=False)["state"]
d = max((a[k].float() - b[k].float()).abs().max().item() for k in a); print(f"ZERO weights: max |diff| vs v13_p3 = {d:.3e} over {len(a)} tensors; keys equal {set(a)==set(b)}")
PYEOF
f4 ilsp_v13_zero
say "CTRL: S's pipeline + steps on v13_p3's own DSM data"
env $TR MAX_LR=5e-5 MAX_STEPS=1331 WIN_DIRS=data/dsm_ft2 X6_DIRS=data/x6 REH="data/fresh2:1.0" \
    nice -n 10 $PY ilwin/train_winner.py ckpt/ilsp_v13_ctrl.pt 1 320 > logs/bugcheck_ctrl_train.log 2>&1 || say "CTRL train failed: $(tail -2 logs/bugcheck_ctrl_train.log)"
grep -E "^\[start\]|^\[save" logs/bugcheck_ctrl_train.log | cut -c1-200 | tee -a $LOG
f4 ilsp_v13_ctrl
say "BUGCHECK DONE"
