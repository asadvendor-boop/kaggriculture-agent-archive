#!/bin/bash
# Sep 26, operator-approved daily refresh + action 1 ("weight imitation data by player strength"):
#  data : Kaggle's daily top games Sep 25 (+ Sep 26 when published) and the top-12 leaders' newest games (their known
#         submissions re-listed + newer submissions found in the rows files) -> seats + v6 features (only new games are
#         built) -> il/data/fresh2 = top-12 games + daily Sep 22-26 (last 3-4 days; held-out games removed) and
#         il/data/dsm_ft2 = every DSM seat (incl. the newest).
#  train: fine-tune the fresh multi-style 10M (fr_t_s0) 2 passes over fresh2 at 3e-4, then the DSM fine-tune (2 x dsm_ft2
#         at 2e-4) - once unweighted (u) and once with rating weights (w), same seed, so the pair differs only by the
#         weights; a second seed pair when GPU memory allows.
#  bench: the 464 ladder worlds (team 10) and the held-out fresh leader benchmark, paired against fr_t_dsm_s0 (429).
cd /Users/asad.ali/kaggriculture
PY=/venv/main/bin/python
L=il/logs/refresh.log
log() { echo "[$(date -u +%H:%M) UTC] $*" | tee -a $L; }
log "refresh starts"
mkdir -p data/kaggle_daily
for d in 2026-09-25 2026-09-26; do
  z=data/kaggle_daily/kaggriculture-episodes-$d.zip
  for a in 1 2 3; do
    $PY -c "import zipfile, sys; sys.exit(0 if zipfile.is_zipfile('$z') else 1)" 2>/dev/null && break
    curl -s -L --retry 3 --retry-delay 30 -o $z https://www.kaggle.com/api/v1/datasets/download/kaggle/kaggriculture-episodes-$d
    sleep 15
  done
  if $PY -c "import zipfile, sys; sys.exit(0 if zipfile.is_zipfile('$z') else 1)" 2>/dev/null; then
    log "$d zip $(du -h $z | cut -f1)"; $PY il/live/srv_import_day.py $z $d data/tapes_kd/$d 6 2>&1 | tail -2 | tee -a $L
  else
    log "$d: no daily dataset yet"; rm -f $z
  fi
done
THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top12_b.log 2>&1
log "top-12 re-listed: $(tail -1 il/logs/fetch_top12_b.log)"
$PY il/live/srv_discover_subs.py il/logs/new_subs.json 2>&1 | tee -a $L
if [ "$($PY -c "import json; print(sum(len(v) for v in json.load(open('il/logs/new_subs.json')).values()))")" -gt 0 ]; then
  SUBS_JSON=il/logs/new_subs.json THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top12_c.log 2>&1
  log "newer leader submissions fetched: $(tail -1 il/logs/fetch_top12_c.log)"
fi
P="data/tapes_top12"; for d in 2026-09-26 2026-09-25 2026-09-24 2026-09-23 2026-09-22; do [ -f data/tapes_kd/$d/index.json ] && P="$P data/tapes_kd/$d"; done
build() {   # $1 tape dirs  $2 workers  $3 nice  $4 log tag   (= srv_fresh_data.sh; existing seats are skipped)
  for w in $(seq 0 $(($2 - 1))); do nice -n $3 $PY il/build_dataset3.py il/data/fresh_all $w $2 $1 > il/logs/build_$4_w$w.log 2>&1 & done; wait
  log "$4 seats written: $(tail -q -n1 il/logs/build_$4_w*.log | awk '{s+=$8} END {print s}'), non-exact/failed: $(tail -q -n1 il/logs/build_$4_w*.log | awk '{s+=$10} END {print s}')"
  for w in $(seq 0 $(($2 - 1))); do TAPES=$(echo $1 | tr ' ' ':') SEATS=il/data/fresh_all nice -n $3 $PY il/live/srv_build_x6.py $w $2 > il/logs/x6_$4_w$w.log 2>&1 & done; wait
  log "$4 v6 features: $(tail -q -n1 il/logs/x6_$4_w*.log | awk '{s+=$5} END {print s}') new, missing tape $(tail -q -n1 il/logs/x6_$4_w*.log | awk '{s+=$8} END {print s}')"
}
build "$P" ${NB:-6} 10 fresh2
$PY il/live/srv_fresh_split.py fresh2 $P 2>&1 | tee -a $L
$PY il/live/srv_build_dsm.py dsm_ft2 2>&1 | tee -a $L
TEAM=0 $PY il/live/srv_build_dsm.py boey_ft2 2>&1 | tee -a $L
SUBS_JSON=il/logs/new_subs.json $PY il/live/srv_rating_w.py il/data/fresh2 il/data/rating_w_fresh2.json $P 2>&1 | tee -a $L
log "REFRESH DATA READY: fresh2 $(ls il/data/fresh2 | wc -l) seats, dsm_ft2 $(ls il/data/dsm_ft2 | wc -l), boey_ft2 $(ls il/data/boey_ft2 | wc -l) seats"
cd il
source live/srv_cpu.sh
run() {  # $1 tag  $2 seed  $3 extra env (or -): refresh pass, then the DSM (team 10) and Boey (team 0) fine-tunes
  local E=""; [ "$3" != "-" ] && E="$3"
  env SEED=$2 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=3e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
    INIT=ckpt/fr_t_s0.pt $E $PY train6.py ckpt/rf_$1_s$2.pt 2 256 data/fresh2 > logs/train_rf_$1_s$2.log 2>&1
  echo "[$(date -u +%H:%M) UTC] refresh pass done $1 s$2: $(grep 'best val loss' logs/train_rf_$1_s$2.log)" | tee -a logs/refresh.log
  for st in dsm:dsm_ft2 boey:boey_ft2; do
    [ -f ckpt/rf_$1_s$2.pt ] && env SEED=$2 BLOCKS=8 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 \
      INIT=ckpt/rf_$1_s$2.pt $PY train6.py ckpt/rf_$1_${st%%:*}_s$2.pt 2 256 data/${st#*:} > logs/train_rf_$1_${st%%:*}_s$2.log 2>&1
    echo "[$(date -u +%H:%M) UTC] refresh ${st%%:*} FT done $1 s$2: $(grep 'best val loss' logs/train_rf_$1_${st%%:*}_s$2.log)" | tee -a logs/refresh.log
  done
  touch logs/rf_$1_s$2.finished
}
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
RW="RATING_W=/Users/asad.ali/kaggriculture/il/data/rating_w_fresh2.json"
for job in "u 0 -" "w 0 $RW"; do
  until [ "$(gpu_free)" -ge 9000 ]; do sleep 60; done
  set -- $job; echo "[$(date -u +%H:%M) UTC] start refresh $1 s$2 (GPU free $(gpu_free) MiB, CPU free $(cpu_free))" | tee -a logs/refresh.log
  run $1 $2 "$3" &
  sleep 180
done
# style check while training (Boey is #1 today): the multi-style model as Boey (0) vs as DSM (10) on the 464 worlds
bench_gov ladder ckpt/fr_t_s0.pt:0:0.5 ckpt/fr_t_s0.pt:10:0.5
wait
bench_gov ladder $(for k in u w; do for st in dsm:10 boey:0; do c=ckpt/rf_${k}_${st%%:*}_s0.pt; [ -f $c ] && echo $c:${st#*:}:0.5; done; done)
for k in u w; do
  f=logs/ladder_rf_${k}_dsm_s0_t10_r3_tau0.5_test.json; [ -f $f ] && $PY live/pair_cmp.py logs/ladder_fr_t_dsm_s0_t10_r3_tau0.5_test.json $f | tee -a logs/refresh.log
  f=logs/ladder_rf_${k}_boey_s0_t0_r3_tau0.5_test.json; [ -f $f ] && $PY live/pair_cmp.py logs/ladder_fr_t_dsm_s0_t10_r3_tau0.5_test.json $f | tee -a logs/refresh.log
done
bench_gov top:10 $(for k in u w; do [ -f ckpt/rf_${k}_dsm_s0.pt ] && echo ckpt/rf_${k}_dsm_s0.pt; done)
bench_gov top:0 $(for k in u w; do [ -f ckpt/rf_${k}_boey_s0.pt ] && echo ckpt/rf_${k}_boey_s0.pt; done)
if [ -f live/top_worlds_fresh2_300.json ]; then          # today's top-10 benchmark: live models + refresh models
  bench_gov top2:10 ckpt/fr_t_dsm_s0.pt ckpt/rl2_it24.pt $(for k in u w; do [ -f ckpt/rf_${k}_dsm_s0.pt ] && echo ckpt/rf_${k}_dsm_s0.pt; done)
  bench_gov top2:0 $(for k in u w; do [ -f ckpt/rf_${k}_boey_s0.pt ] && echo ckpt/rf_${k}_boey_s0.pt; done)
fi
echo "[$(date -u +%H:%M) UTC] REFRESH CHAIN DONE" | tee -a logs/refresh.log
