#!/bin/bash
# Sep 27 ~14:27 UTC: replaces the sprint's first-come bencher. Leader tests (top:10 + top2:10, SE guard on) of the
# finished models in PRIORITY order - more fresh2 passes first (p4 > p3 > p2 = soups > p1 > DSM-only fine-tunes >
# v10/v11), so the most promising variants are tested before the 2-hour mark. A model already tested ('== <name>' in
# /root/il_bench.log) is skipped. Paired vs BOTH active submissions: NEURAL_20MW5_SE (g20_w5_dsm_s0) and NEURAL_20MW_SE
# (g20_w_dsm_s0, the one a new upload would retire).
exec 8>/tmp/srv_il_bencher2.lock; flock -n 8 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
source live/srv_cpu.sh; export CPU_BUDGET=${CPU_BUDGET_B:-110} BENCH_CAP=${BENCH_CAP_B:-100} SE_GUARD=1
BQ=/root/il_bench_queue.txt
# ~14:30 UTC: v3_ft3 (3 DSM epochs) is the first model to beat NEURAL_20MW5_SE on the 464 -> the 3-5 epoch fine-tunes
# (its replication v9_ft3s2, ft4, ft5) share the top priority with the longer fresh2 runs
# ~14:52 UTC: 4 DSM epochs (v22_ft4) lost to 3 -> ft4/ft5 drop to the end; the p2/p3 + 3-epoch combos (srv_il_extra.sh) rank high
prio() { case $1 in *_p4*) echo 6;; *_ft4*|*_ft5*) echo 1;; *_ft3*|*_p3*) echo 5;; *soup*|*_p2*) echo 4;; *v10_*|*v11_*) echo 0;; *_p1*) echo 2;; *) echo 1;; esac; }
while true; do
  best=""; bp=-1
  while read -r m; do
    [ -z "$m" ] && continue
    grep -qx "== $m" /root/il_bench.log && continue
    [ -f ckpt/$m.pt ] || continue
    p=$(prio $m); [ "$p" -gt "$bp" ] && { best=$m; bp=$p; }
  done < $BQ
  if [ -z "$best" ]; then
    [ "$(grep -c 'queue empty' /root/il_sprint.log)" -ge 8 ] && break
    sleep 20; continue
  fi
  TAGX=_se bench_gov top:10 ckpt/$best.pt > logs/ilsp_bench_${best}_top.log 2>&1
  TAGX=_se bench_gov top2:10 ckpt/$best.pt > logs/ilsp_bench_${best}_top2.log 2>&1
  { echo "== $best"; grep -h -E " n (138|122) " logs/ilsp_bench_${best}_top*.log | cut -c1-120
    for d in fresh fresh2; do
      $PY live/pair_cmp.py logs/top_${d}_g20_w5_dsm_s0_t10_se.json logs/top_${d}_${best}_t10_se.json | sed 's/^/  vs 20MW5_SE /' | cut -c1-200
      $PY live/pair_cmp.py logs/top_${d}_g20_w_dsm_s0_t10_se.json logs/top_${d}_${best}_t10_se.json | sed 's/^/  vs 20MW_SE  /' | cut -c1-200
    done; } >> /root/il_bench.log
done
echo "[$(date -u +%H:%M) UTC] leader tests done for all models" >> /root/il_bench.log
