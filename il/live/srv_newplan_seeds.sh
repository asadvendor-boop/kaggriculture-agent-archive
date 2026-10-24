#!/bin/bash
# Sep 28 ~05:45 PKT: is ilsp_v13_newplan (seed 0) a lucky seed? newplan3 (+110 newest seats) fell 263 -> 244 on the rulers
# (fresh 121 -> 110) while staying close on the live losses (42 vs 44), and fine-tune siblings have swung +-15 leader wins
# before. Train seeds 1 and 2 of the exact newplan recipe (same data, 1 epoch at 1e-4 from v13_p3) and the uniform soup of
# seeds 0-2 (live/soup.py), then test each with SELL_ORDER=held SE_GUARD=0 paired vs npnoseh (newplan seed 0), queued after
# the other rule tests. Log /root/newplan.log
exec 8>/tmp/srv_newplan_seeds.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; LOG=/root/newplan.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $LOG; }
for s in 1 2; do
  while pgrep -f "python train6.py" > /dev/null; do sleep 60; done
  say "fine-tune: newplan seed $s (v13_p3 + 1 epoch of newplan at 1e-4)"
  env SEED=$s MAX_LR=1e-4 INIT=ckpt/ilsp_v13_p3.pt BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1 \
    $PY train6.py ckpt/ilsp_v13_newplan_s$s.pt 1 320 data/newplan > logs/ilsp_v13_newplan_s$s.log 2>&1
  say "NEWPLAN S$s DONE: $(grep 'best val loss' logs/ilsp_v13_newplan_s$s.log | tail -1)"
done
$PY live/soup.py ckpt/ilsp_v13_newplan_soup3.pt ckpt/ilsp_v13_newplan.pt ckpt/ilsp_v13_newplan_s1.pt ckpt/ilsp_v13_newplan_s2.pt >> $LOG 2>&1
say "NEWPLAN SOUP3 written"
prev="RULE n3npnoseh done"
for m in ilsp_v13_newplan_s1 ilsp_v13_newplan_s2 ilsp_v13_newplan_soup3; do
  t=${m#ilsp_v13_newplan_}nose
  CK=$m BASECK=ilsp_v13_newplan BASET=npnoseh WAITFOR="$prev" setsid -f bash live/srv_ruletest.sh $t SELL_ORDER=held SE_GUARD=0 > /root/rt_$t.out 2>&1 < /dev/null 8>&-
  prev="RULE $t done"; sleep 2
done
say "seed tests queued: s1nose, s2nose, soup3nose (after n3npnoseh)"
