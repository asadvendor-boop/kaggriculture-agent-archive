#!/bin/bash
# Sep 27 IL sprint finale (8x H100 box). Output goes to /root/il_bench.log (the Mac monitor shows it).
#   srv_il_final.sh 464 <model> ...     464 ladder worlds WITH the SE guard for each model (ckpt/<model>.pt), paired vs
#                                       NEURAL_20MW5_SE (g20_w5_dsm_s0 + SE: 450 / 2400+ 212); ~3 min each at 90 workers
#   srv_il_final.sh pkg <model> <NAME>  package/<NAME>/main.py exactly like NEURAL_20MW5_SE (base64 weights, SE guard,
#                                       tau 0.5, team 10) + <NAME>.tar.gz, then srv_pkgcheck.sh (12-world parity vs the
#                                       model's 464 SE rows, official kaggle_environments check vs V94)
exec 7>/tmp/srv_il_final.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python
source live/srv_cpu.sh; export CPU_BUDGET=${CPU_BUDGET_F:-110} BENCH_CAP=${BENCH_CAP_F:-100}
BASE=logs/ladder_g20_w5_dsm_s0_t10_r3_tau0.5_se_test.json
mode=$1; shift
if [ "$mode" = 464 ]; then
  for m in "$@"; do
    SE_GUARD=1 TAGX=_r3_tau0.5_se bench_gov ladder ckpt/$m.pt:10:0.5 > logs/ilsp_bench_${m}_464.log 2>&1
    { echo "== 464 $m"; grep -E " n 464 " logs/ilsp_bench_${m}_464.log | cut -c1-200
      $PY live/pair_cmp.py $BASE logs/ladder_${m}_t10_r3_tau0.5_se_test.json | sed 's/^/  vs 20MW5_SE 464 /' | cut -c1-260; } >> /root/il_bench.log
  done
elif [ "$mode" = pkg ]; then
  m=$1; N=$2; mkdir -p $K/package/$N
  B64=1 RULES="SE_GUARD=1" $PY export.py ckpt/$m.pt $K/package/$N/main.py 0.5 10 2>&1 | grep -v -i warn | tail -3
  tar czf $K/package/$N.tar.gz -C $K/package/$N main.py
  echo "[$(date -u +%H:%M) UTC] PKG $N from $m: $(ls -l $K/package/$N.tar.gz | awk '{print $5}') bytes, main.py sha256 $(sha256sum $K/package/$N/main.py | cut -c1-16)" >> /root/il_bench.log
  bash live/srv_pkgcheck.sh $N logs/ladder_${m}_t10_r3_tau0.5_se_test.json > logs/pkgcheck_$N.log 2>&1
  grep -E -i "parity|match|exact|mismatch|official|PKGCHECK|error|FAIL" logs/pkgcheck_$N.log | cut -c1-200 | sed "s/^/  [pkg $N] /" >> /root/il_bench.log
fi
