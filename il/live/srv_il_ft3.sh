#!/bin/bash
# Sep 27 ~15:01 UTC: one 3-epoch DSM fine-tune on a given GPU (idle GPUs after the sprint queue emptied).
#   srv_il_ft3.sh <gpu> <name> <init ckpt> [seed]   -> ckpt/ilsp_<name>.pt, logs/ilsp_<name>.log, bench queue
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1"
g=$1; name=$2; init=$3; seed=${4:-0}; log=logs/ilsp_$name.log
echo "[$(date -u +%H:%M) UTC] gpu$g $name: DSM fine-tune 3 epoch(s) lr 2e-4 seed $seed from $init" >> $log
env CUDA_VISIBLE_DEVICES=$g SEED=$seed MAX_LR=2e-4 INIT=$init $C $PY train6.py ckpt/ilsp_${name}.pt 3 320 data/dsm_ft2 >> $log 2>&1
echo "[$(date -u +%H:%M) UTC] gpu$g $name DONE: $(grep 'best val loss' $log | tail -1)" | tee -a $log >> /root/il_sprint.log
[ -f ckpt/ilsp_${name}.pt ] && echo "ilsp_${name}" >> /root/il_bench_queue.txt
