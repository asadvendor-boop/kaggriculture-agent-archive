#!/bin/bash
# Sep 27 ~14:52 UTC (20 min before the last sprint job ends; the sprint's job queue is empty, GPUs are freeing up):
# v3_ft3 (3 DSM epochs) is the best model and extra fresh2 passes (v2_p2) the runner-up -> combine the two: 3-epoch DSM
# fine-tunes from the pass-stage checkpoints (ckpt/ilsp_<v>_p.pt, i.e. before their 2-epoch fine-tune). ~7 min each.
# One worker per GPU: waits until no train6 job uses its GPU, then takes the next extra job; finished models go to the
# bench queue (srv_il_bencher2 ranks '*_ft3*' high).
exec 8>/tmp/srv_il_extra.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1"
Q=/root/il_extra_jobs.txt
cat > $Q <<JOBS
v25_p2_ft3 ckpt/ilsp_v2_p2_p.pt 3 2e-4 0
v27_p3_ft3 ckpt/ilsp_v13_p3_p.pt 3 2e-4 0
v26_p2s1_ft3 ckpt/ilsp_v14_p2s1_p.pt 3 2e-4 1
v28_p2sq_ft3 ckpt/ilsp_v15_p2sq_p.pt 3 2e-4 0
JOBS
gpu_busy() { local p; for p in $(pgrep -f "python train6.py"); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
worker() {
  local g=$1 line name init ep lr seed log
  while true; do
    while gpu_busy $g; do sleep 15; done
    line=$(flock /tmp/il_extra.qlock bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
    [ -z "$line" ] && break
    set -- $line; name=$1; init=$2; ep=$3; lr=$4; seed=$5; log=logs/ilsp_$name.log
    echo "[$(date -u +%H:%M) UTC] gpu$g $name: DSM fine-tune $ep epoch(s) lr $lr seed $seed from $init" >> $log
    env CUDA_VISIBLE_DEVICES=$g SEED=$seed MAX_LR=$lr INIT=$init $C $PY train6.py ckpt/ilsp_${name}.pt $ep 320 data/dsm_ft2 >> $log 2>&1
    echo "[$(date -u +%H:%M) UTC] gpu$g $name DONE: $(grep 'best val loss' $log | tail -1)" | tee -a $log >> /root/il_sprint.log
    [ -f ckpt/ilsp_${name}.pt ] && echo "ilsp_${name}" >> /root/il_bench_queue.txt
  done
}
for g in 0 1 2 3 4 5 6 7; do worker $g & sleep 1; done
wait
echo "[$(date -u +%H:%M) UTC] extra fine-tunes done" >> /root/il_sprint.log
