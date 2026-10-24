#!/bin/bash
# Sep 27 ~19:00 PKT (operator: pivot the 8x H100 to imitation models, all 8 GPUs busy, done in ~2 h): fine-tune variants
# of our best 20M (g20_w5_s0 = 5-pass weighted, before the DSM fine-tune; g20_w5_dsm_s0 = the live NEURAL_20MW5_SE)
# through a JOB QUEUE: 8 GPU workers, longest jobs first; a GPU that finishes takes the next job, so none idles.
# Every finished model is benchmarked on the CPUs WITH the SE guard (as it would ship): leader tests first
# (top:10, top2:10), then the 464 ladder worlds; paired vs NEURAL_20MW5_SE (g20_w5_dsm_s0 + SE).
exec 8>/tmp/srv_il_sprint.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY - <<'PYX'
import json; w = json.load(open("data/rating_w_fresh2.json")); json.dump({k: v * v for k, v in w.items()}, open("data/rating_w_fresh2_sq.json", "w"))
print("steep rating weights written")
PYX
C="BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1"
RW=$K/il/data/rating_w_fresh2.json; RWS=$K/il/data/rating_w_fresh2_sq.json
Q=/root/il_jobs.txt; : > $Q
# name | pass stage (init, passes, lr, seed, weights) | DSM fine-tune stage (epochs, lr, seed)     longest first
cat >> $Q <<JOBS
v2_p2|ckpt/g20_w5_s0.pt 2 1.5e-4 0 $RW|2 2e-4 0
v1_p1|ckpt/g20_w5_s0.pt 1 1e-4 0 $RW|2 2e-4 0
v5_p1s1|ckpt/g20_w5_s0.pt 1 1e-4 1 $RW|2 2e-4 1
v7_p1sq|ckpt/g20_w5_s0.pt 1 1e-4 0 $RWS|2 2e-4 0
v8_live_p1|ckpt/g20_w5_dsm_s0.pt 1 5e-5 0 $RW|1 1e-4 0
v3_ft3|-|3 2e-4 0
v6_ft2s1|-|2 2e-4 1
v4_live_ft1|@|1 1e-4 0
v9_ft3s2|-|3 2e-4 2
v10_ft2lr1|-|2 1e-4 0
v11_live_ft1s1|@|1 1e-4 1
v12_p1s2|ckpt/g20_w5_s0.pt 1 1e-4 2 $RW|2 2e-4 2
JOBS
BQ=/root/il_bench_queue.txt; : > $BQ
run_job() {   # run_job <gpu> <line>
  local g=$1 name=${2%%|*} rest=${2#*|}; local p1=${rest%%|*} ft=${rest#*|}
  local init=ckpt/g20_w5_s0.pt log=logs/ilsp_${name}.log
  if [ "$p1" = "@" ]; then init=ckpt/g20_w5_dsm_s0.pt
  elif [ "$p1" != "-" ]; then
    set -- $p1
    echo "[$(date -u +%H:%M) UTC] gpu$g $name: $2 pass(es) from $1 lr $3 seed $4" >> $log
    env CUDA_VISIBLE_DEVICES=$g SEED=$4 MAX_LR=$3 RATING_W=$5 INIT=$1 $C $PY train6.py ckpt/ilsp_${name}_p.pt $2 320 data/fresh2 >> $log 2>&1
    init=ckpt/ilsp_${name}_p.pt
  fi
  set -- $ft
  echo "[$(date -u +%H:%M) UTC] gpu$g $name: DSM fine-tune $1 epoch(s) lr $2 seed $3 from $init" >> $log
  env CUDA_VISIBLE_DEVICES=$g SEED=$3 MAX_LR=$2 INIT=$init $C $PY train6.py ckpt/ilsp_${name}.pt $1 320 data/dsm_ft2 >> $log 2>&1
  echo "[$(date -u +%H:%M) UTC] gpu$g $name DONE: $(grep 'best val loss' $log | tail -1)" | tee -a $log >> /root/il_sprint.log
  [ -f ckpt/ilsp_${name}.pt ] && echo "ilsp_${name}" >> $BQ
}
worker() {    # one per GPU: pop jobs until the queue is empty
  local g=$1
  while true; do
    local line; line=$(flock /tmp/il_jobs.qlock bash -c "l=\$(head -1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
    [ -z "$line" ] && break
    run_job $g "$line"
  done
  echo "[$(date -u +%H:%M) UTC] gpu$g: queue empty" >> /root/il_sprint.log
}
bencher() {   # CPU side: leader tests first, then the 464, for every finished model (SE guard on, as shipped)
  source live/srv_cpu.sh; export CPU_BUDGET=100 BENCH_CAP=90 SE_GUARD=1
  local done_n=0
  while true; do
    local m; m=$(sed -n "$((done_n + 1))p" $BQ)
    if [ -z "$m" ]; then
      [ "$(grep -c 'queue empty' /root/il_sprint.log)" -ge 8 ] && [ "$done_n" -ge "$(wc -l < $BQ)" ] && break
      sleep 20; continue
    fi
    done_n=$((done_n + 1))
    TAGX=_se bench_gov top:10 ckpt/$m.pt > logs/ilsp_bench_${m}_top.log 2>&1
    TAGX=_se bench_gov top2:10 ckpt/$m.pt > logs/ilsp_bench_${m}_top2.log 2>&1
    { echo "== $m"; grep -h -E " n (138|122) " logs/ilsp_bench_${m}_top*.log | cut -c1-120
      for d in fresh fresh2; do $PY live/pair_cmp.py logs/top_${d}_g20_w5_dsm_s0_t10_se.json logs/top_${d}_${m}_t10_se.json | sed 's/^/  vs 20MW5_SE /' | cut -c1-200; done; } >> /root/il_bench.log
  done
  echo "[$(date -u +%H:%M) UTC] leader tests done for all models" >> /root/il_bench.log
}
echo "[$(date -u +%H:%M) UTC] IL sprint starts: $(wc -l < $Q) jobs on 8 GPUs" >> /root/il_sprint.log
for g in 0 1 2 3 4 5 6 7; do worker $g & sleep 2; done
bencher &
wait
echo "[$(date -u +%H:%M) UTC] IL SPRINT DONE" >> /root/il_sprint.log
