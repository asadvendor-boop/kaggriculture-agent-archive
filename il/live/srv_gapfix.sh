#!/bin/bash
# GAP FIX, then IL (operator, Sep 25: "fix all the gaps before training then do IL and end with RL"). Gaps found by the
# DSM-likeness report + menu check: (1) old data -> no 3rd field on day 10, little fertilizing, few tomatoes: fixed by
# training on the freshest games only (data/fresh); (2) DSM pads 11.7% of its market turns, which the old order menu
# cannot express: fixed by the extended menu with padding heads (train8, menu 8; it lifted the DSM clone 256 -> 312).
# For each fresh-pass checkpoint (1.33M c, 4M g, 10M t; seeds 0/1) as soon as it exists: a menu-8 pass over data/fresh
# (1 epoch, 5e-4), then a DSM+DECEM fine-tune (2 epochs, 2e-4; data/dsmdecem8 = 608 DSM + 605 DECEM seats, Sep 21-25,
# held-out games removed; DECEM plays DSM's plan and is 99.6% expressible), then the 464-world benchmark as DSM (team 10)
# and as DECEM (team 3). CPU governor; this chain keeps logs/fr_pending = its running trainers (governor accounting).
cd /Users/asad.ali/kaggriculture/il
PY=/venv/main/bin/python
source live/srv_cpu.sh
( while true; do pgrep -fc "python train8.py" > logs/fr_pending; sleep 20; done ) &
ACC=$!
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gap() {  # $1 size tag (c/g/t)  $2 seed  $3 width  $4 blocks
  until grep -q "fresh pass done $1 s$2" logs/fresh_train.log 2>/dev/null; do sleep 120; done
  [ -f ckpt/fr_$1_s$2.pt ] || { echo "[$(date -u +%H:%M) UTC] no fresh checkpoint $1 s$2 - skipped"; touch logs/g8_$1_s$2.finished; return; }
  until [ "$(gpu_free)" -ge 7000 ]; do sleep 60; done
  for a in 1 2 3; do
    env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=5e-4 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=2 PREFETCH=1 INIT=ckpt/fr_$1_s$2.pt \
      $PY train8.py ckpt/g8_$1_s$2.pt 1 $3 data/fresh > logs/train_g8_$1_s$2.log 2>&1
    grep -q "best val loss" logs/train_g8_$1_s$2.log && break
    echo "[$(date -u +%H:%M) UTC] menu-8 pass $1 s$2 FAILED (try $a): $(tail -1 logs/train_g8_$1_s$2.log | cut -c1-160)"; sleep 300
  done
  echo "[$(date -u +%H:%M) UTC] menu-8 pass done $1 s$2: $(grep 'best val loss' logs/train_g8_$1_s$2.log)"
  for a in 1 2 3; do
    [ -f ckpt/g8_$1_s$2.pt ] || break
    env SEED=$2 BLOCKS=$4 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 MAX_LR=2e-4 CHUNK=100 AUX_W=0.1 LOAD_THREADS=2 PREFETCH=1 INIT=ckpt/g8_$1_s$2.pt \
      $PY train8.py ckpt/g8_$1_dd_s$2.pt 2 $3 data/dsmdecem8 > logs/train_g8_$1_dd_s$2.log 2>&1
    grep -q "best val loss" logs/train_g8_$1_dd_s$2.log && break
    echo "[$(date -u +%H:%M) UTC] DSM+DECEM FT $1 s$2 FAILED (try $a): $(tail -1 logs/train_g8_$1_dd_s$2.log | cut -c1-160)"; sleep 300
  done
  echo "[$(date -u +%H:%M) UTC] DSM+DECEM FT done $1 s$2: $(grep 'best val loss' logs/train_g8_$1_dd_s$2.log)"
  touch logs/g8_$1_s$2.finished
}
gap c 0 96 5 & sleep 60
gap c 1 96 5 & sleep 60
gap g 0 176 6 & sleep 60
gap g 1 176 6 & sleep 60
gap t 0 256 8 & sleep 60
gap t 1 256 8 &
ALL="c_s0 c_s1 g_s0 g_s1 t_s0 t_s1"; DONE=""
while [ $(echo $DONE | wc -w) -lt 6 ]; do
  for k in $ALL; do
    case " $DONE " in *" $k "*) continue ;; esac
    if [ -f logs/g8_${k%_s*}_s${k#*_s}.finished ]; then
      c=ckpt/g8_${k%_s*}_dd_s${k#*_s}.pt
      [ -f $c ] && bench_gov ladder $c:10:0.5 $c:3:0.5
      DONE="$DONE $k"
    fi
  done
  sleep 120
done
wait $(jobs -p | grep -v "^$ACC$") 2>/dev/null
kill $ACC; rm -f logs/fr_pending
echo "[$(date -u +%H:%M) UTC] GAPFIX CHAIN DONE"
