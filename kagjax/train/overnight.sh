#!/bin/bash
# Overnight self-play run on the SHARED A100 (operator, Sep 26 ~21:30 UTC; deadlines from the coordinator ~21:45 UTC).
# Launch ONCE, detached:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/overnight.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/overnight.out 2>&1 < /dev/null &
# - single instance (flock); waits until the 20M weighted 5-pass imitation chain (il/live/srv_g20w5.sh) has finished its
#   DSM fine-tune (log line "DSM FT done", no train6 g20_w5 process left); if the run cannot start by 23:30 UTC it is
#   skipped (DONE = "SKIPPED");
# - runs OUTSIDE the per-test GPU lock with its own cap: XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 (10 GB), no preallocation,
#   cores 56-63, nice 10; the frozen gate-passing trainer (ppo_gate.py), gate-5 settings, 256 games per batch (needs
#   >= 10.5 GB free GPU memory);
# - ends by 02:00 UTC: the trainer is stopped at 01:48 UTC, the last training state (saved every iteration) becomes
#   final.pt (ppo_live .pt, il/export.py-ready), older checkpoints are pruned to the last 3 (+ final), and
#   overnight/DONE gets the final checkpoint's path.
exec 9>/tmp/kagjax_overnight.lock
flock -n 9 || { echo "[$(date -u +%H:%M) UTC] already running - nothing to do"; exit 0; }
R=/Users/asad.ali/kaggriculture
OUT=$R/kagjax/logs_server/D/overnight
PY=/opt/jaxenv/bin/python
mkdir -p "$OUT"
rm -f "$OUT/DONE"
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*"; }
epoch_at() { date -u -d "$(date -u +%Y-%m-%d) $1" +%s; }        # today (UTC) at HH:MM
now() { date -u +%s; }
START_BY=$(epoch_at 23:30)
STOP_AT=$(( $(epoch_at 01:48) + 86400 ))                          # 01:48 UTC tomorrow
il_running() { ps -eo args | grep -c "^/venv/main/bin/python train6.py ckpt/g20_w5"; }        # read-only check
say "waiting for the imitation chain (il/logs/g20w5.log: DSM FT done); latest start 23:30 UTC, stop at 01:48 UTC"
until grep -q "DSM FT done" $R/il/logs/g20w5.log 2>/dev/null && [ "$(il_running)" -eq 0 ]; do
  if [ "$(now)" -ge "$START_BY" ]; then say "not started by 23:30 UTC - SKIPPED"; echo "SKIPPED" > "$OUT/DONE"; exit 0; fi
  sleep 60
done
say "imitation chain finished: $(grep 'DSM FT done' $R/il/logs/g20w5.log)"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until [ "$(free_mb)" -ge 10752 ]; do
  if [ "$(now)" -ge "$START_BY" ]; then say "GPU memory short until 23:30 UTC - SKIPPED"; echo "SKIPPED" > "$OUT/DONE"; exit 0; fi
  say "GPU free $(free_mb) MiB < 10.5 GB, waiting"; sleep 60
done
say "GPU free $(free_mb) MiB; compute apps:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONPATH=$R:$PYTHONPATH
export KAGJAX_BACKEND=real INIT=$R/il/ckpt/g20_w_dsm_s0.pt IL=$R/il/ckpt/g20_w_s0.pt TEAM=10
export B=256 K=4 SEG=720 MB=256 EPOCHS=2 LR=5e-6 VLR=1e-4 BETA=0.05 VWARM=1 T=0.2 KSHARP=10 TAU=0.5
export OPPS=LATEST=2,SELF=2,IL=4 IL_TEAMS=10:3,5:2,3:2,4:2,0:2 SNAP_EVERY=4 SELF_MAX=6
export ROLLOUT_PRECISION=default UPDATE_PRECISION=default EVAL_PRECISION=highest OLD_LOGP=behavior DIAG_EVERY=10
export EVAL_EVERY=4 EVAL_N=128 EVAL_OPPS=SNAP0,IL@10,IL@0 CKPT_EVERY=1 PT_EVERY=0 PT_EVERY_MIN=60 ITERS=500 KL_MAX=1.0 SEED=1
export RUN_DIR=$OUT
SECS=$(( STOP_AT - $(now) ))
say "starting: B=$B K=$K SEG=$SEG MB=$MB EPOCHS=$EPOCHS LR=$LR BETA=$BETA INIT=$INIT IL=$IL -> $OUT (runs ${SECS}s, until 01:48 UTC)"
cd $R
timeout --signal=TERM --kill-after=60 ${SECS}s taskset -c 56-63 nice -n 10 $PY $R/kagjax/train/ppo_gate.py
say "trainer ended (exit $?)"
# ---- final checkpoint: the newest training state -> final.pt; prune older .pt to the last 3; DONE marker
LAST=$(ls $OUT/state_*.npz 2>/dev/null | sed 's/.*state_\([0-9]*\)\.npz/\1/' | sort -n | tail -1)
if [ -n "$LAST" ]; then
  taskset -c 56-63 nice -n 10 env JAX_PLATFORMS=cpu $PY $R/kagjax/train/ckpt_io.py to_pt $OUT/state_$LAST.npz $OUT/final.pt
  say "final.pt from state_$LAST.npz (iteration $LAST)"
  KEEP=3; [ -f $OUT/best.pt ] && KEEP=2                        # at most 3 checkpoints besides final.pt (best.pt counts)
  for f in $(ls $OUT/pt/it*.pt 2>/dev/null | sed 's/.*it\([0-9]*\)\.pt/\1/' | sort -n | head -n -$KEEP); do rm -f $OUT/pt/it$f.pt; done
  ls $OUT/state_*.npz | sed 's/.*state_\([0-9]*\)\.npz/\1/' | sort -n | head -n -1 | while read s; do rm -f $OUT/state_$s.npz; done
  echo "$OUT/final.pt" > "$OUT/DONE"
  say "DONE: $(cat $OUT/DONE); kept: $(ls $OUT/pt/ 2>/dev/null | tr '\n' ' ') best.pt=$(ls $OUT/best.pt 2>/dev/null) state_$LAST.npz"
else
  say "no training state was written - no final checkpoint"; echo "FAILED: no training state" > "$OUT/DONE"
fi
