#!/bin/bash
# Sep 28: build an upload candidate with any rule set and check it (never uploads; the operator uploads).
# Usage: srv_mkpkg.sh <ckpt name> <package name> <RULES> <RULES_STR or -> <464 benchmark json of the same model + rules>
#   e.g. srv_mkpkg.sh ilsp_v13_p3 NEURAL_20MW8_SE_M4 "SE_GUARD=1,MELON_WAVE=4" - logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_dsm_m4_test.json
# export (base64 blob, tau 0.5, team 10) -> package/<name>.tar.gz -> srv_pkgcheck.sh (12-world exact parity vs the 464 rows +
# the official kaggle_environments loader, both seats). Summary lines go to /root/a100_bench.log.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
m=$1; N=$2; R=$3; RS=$4; J=$5
[ "$RS" = "-" ] && RS=""
mkdir -p $K/package/$N
B64=1 RULES="$R" RULES_STR="$RS" $PY export.py ckpt/$m.pt $K/package/$N/main.py 0.5 10 2>&1 | grep -v -i warn | tail -4 > logs/mkpkg_$N.log
tar czf $K/package/$N.tar.gz -C $K/package/$N main.py
echo "[$(date -u +%H:%M) UTC] PKG $N from $m (RULES $R${RS:+ | $RS}): $(ls -l $K/package/$N.tar.gz | awk '{print $5}') bytes, main.py sha256 $(sha256sum $K/package/$N/main.py | cut -c1-16)" >> $L
grep -E "rule layer|wrote" logs/mkpkg_$N.log | sed "s/^/  [pkg $N] /" >> $L
bash live/srv_pkgcheck.sh $N $J > logs/pkgcheck_$N.log 2>&1
grep -E -i "parity|match|exact|mismatch|official|PKGCHECK|error|FAIL" logs/pkgcheck_$N.log | grep -v "EXACT$" | cut -c1-200 | sed "s/^/  [pkg $N] /" >> $L
