#!/bin/bash
# Sep 28 ~03:30 PKT: the same new-plan fine-tune (il/data/newplan: DSM + M&M&P&Q seats of Sep 26-27 + 2,000 anchors,
# 1 epoch at 1e-4) on top of n3 (fresh3 3-pass recipe incl. the Sep 27 leader games) while the GPU is otherwise idle
# -> ckpt/ilsp_n3_newplan.pt -> srv_a100_eval.sh (queued). Log /root/newplan.log
exec 8>/tmp/srv_newplan_n3.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; LOG=/root/newplan.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
while pgrep -f "python train6.py" > /dev/null; do sleep 60; done
say "fine-tune: n3 + 1 epoch of newplan at 1e-4"
env SEED=0 MAX_LR=1e-4 INIT=ckpt/ilsp_n3.pt BLOCKS=10 N_TEAMS=50 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 \
  $PY train6.py ckpt/ilsp_n3_newplan.pt 1 320 data/newplan > logs/ilsp_n3_newplan.log 2>&1
say "N3 NEWPLAN DONE: $(grep 'best val loss' logs/ilsp_n3_newplan.log | tail -1)"
setsid -f bash live/srv_a100_eval.sh ilsp_n3_newplan > /dev/null 2>&1 < /dev/null 8>&-
