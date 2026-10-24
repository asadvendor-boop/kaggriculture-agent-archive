#!/bin/bash
# The 4 options on FRESH data (operator, Sep 25), aimed at DSM ("if we need to reach the top, we must imitate DSM best"):
# a warm start, a pass at 5e-4 over data/fresh (top-12 current submissions + daily top games Sep 21-24, held-out games
# removed), then a DSM fine-tune (2 epochs at 2e-4 on data/dsm_ft = every DSM seat we have, held-out games removed),
# played in DSM's style (team 10):
#   c = 1.33M from bc_v6    g = 4M (grown copy of bc_v6)    t = 10M (grown copy)    - seeds 0 and 1
#   x = 1.33M FROM SCRATCH (control, twice the passes at 2e-3)                      - seed 0
# The fresh set is smaller than the union, so the number of passes is set to give about the union pass's step count.
# Jobs start as soon as the data is ready (longest first), each when >= 7 GB of GPU memory is free; 3 tries per stage.
# Benchmarks through the CPU governor (srv_cpu.sh: >= 2 cores stay free, one benchmark at a time): the 464 ladder worlds
# as each model finishes, then the held-out fresh leader benchmark (top_worlds_fresh_300.json) for all of them plus the
# two best earlier models (Vadim style, team 5).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
echo 7 > logs/fr_pending
until grep -q "FRESH READY" logs/fresh_data.log 2>/dev/null; do sleep 60; done
S=$(ls data/fresh | wc -l)
E=$($PY -c "print(max(1, min(4, round(29588 / max(1, $S)))))"); EX=$((2 * E))
$PY live/srv_build_dsm.py dsm_ft
echo "[$(date -u +%H:%M) UTC] fresh training starts: $S seats, $E passes (scratch control $EX); DSM fine-tune set $(ls data/dsm_ft | wc -l) seats"
run() {  # $1 tag  $2 seed  $3 width  $4 blocks  $5 init (none = from scratch)  $6 passes  $7 max lr
  if [ "$5" = none ]; then I=""; else I="INIT=$5"; fi
  for a in 1 2 3; do
    env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=$7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 $I \
      $PY train6.py ckpt/fr_$1_s$2.pt $6 $3 data/fresh > logs/train_fr_$1_s$2.log 2>&1
    grep -q "best val loss" logs/train_fr_$1_s$2.log && break
    echo "[$(date -u +%H:%M) UTC] $1 s$2 pass FAILED (try $a): $(tail -1 logs/train_fr_$1_s$2.log | cut -c1-160)"; sleep 300
  done
  echo "[$(date -u +%H:%M) UTC] fresh pass done $1 s$2: $(grep 'best val loss' logs/train_fr_$1_s$2.log)"
  for a in 1 2 3; do
    [ -f ckpt/fr_$1_s$2.pt ] || break
    env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=3 PREFETCH=1 INIT=ckpt/fr_$1_s$2.pt \
      $PY train6.py ckpt/fr_$1_dsm_s$2.pt 2 $3 data/dsm_ft > logs/train_fr_$1_dsm_s$2.log 2>&1
    grep -q "best val loss" logs/train_fr_$1_dsm_s$2.log && break
    echo "[$(date -u +%H:%M) UTC] $1 s$2 DSM FT FAILED (try $a): $(tail -1 logs/train_fr_$1_dsm_s$2.log | cut -c1-160)"; sleep 300
  done
  echo "[$(date -u +%H:%M) UTC] DSM FT done $1 s$2: $(grep 'best val loss' logs/train_fr_$1_dsm_s$2.log)"
  touch logs/fr_$1_s$2.finished
}
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
LEFT=7
for job in "t 0 256 8 ckpt/grow_v6_256x8.pt $E 5e-4" "t 1 256 8 ckpt/grow_v6_256x8.pt $E 5e-4" \
           "g 0 176 6 ckpt/grow_v6_176x6.pt $E 5e-4" "g 1 176 6 ckpt/grow_v6_176x6.pt $E 5e-4" \
           "x 0 96 5 none $EX 2e-3" "c 0 96 5 ckpt/bc_v6.pt $E 5e-4" "c 1 96 5 ckpt/bc_v6.pt $E 5e-4"; do
  until [ "$(gpu_free)" -ge 7000 ]; do sleep 60; done
  echo "[$(date -u +%H:%M) UTC] start $job (GPU free $(gpu_free) MiB, CPU free $(cpu_free))"
  run $job &
  LEFT=$((LEFT - 1)); echo $LEFT > logs/fr_pending
  sleep 150
done
ALL="t_s0 t_s1 g_s0 g_s1 x_s0 c_s0 c_s1"; DONE=""
while [ $(echo $DONE | wc -w) -lt 7 ]; do
  READY=""
  for k in $ALL; do
    case " $DONE " in *" $k "*) continue ;; esac
    [ -f logs/fr_${k%_s*}_s${k#*_s}.finished ] && READY="$READY $k"
  done
  if [ -n "$READY" ]; then
    SPECS=$(for k in $READY; do c=ckpt/fr_${k%_s*}_dsm_s${k#*_s}.pt; [ -f $c ] && echo $c:10:0.5; done)
    [ -n "$SPECS" ] && bench_gov ladder $SPECS
    DONE="$DONE $READY"
  else
    sleep 120
  fi
done
wait
rm -f logs/fr_pending
bench_gov top:10 $(for k in $ALL; do c=ckpt/fr_${k%_s*}_dsm_s${k#*_s}.pt; [ -f $c ] && echo $c; done)
bench_gov top:5 ckpt/bc_v6_vadim.pt ckpt/s2_grown_s4.pt
echo "[$(date -u +%H:%M) UTC] FRESH CHAIN DONE"
