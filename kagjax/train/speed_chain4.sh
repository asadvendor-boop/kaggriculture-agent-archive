#!/bin/bash
# KL-anchor subsample experiment again (Sep 26 ~23:10 UTC): the first try (queued by speed_chain.sh) stalled in the
# allocator (gradient chunks of 1,024 samples need 4.2 GiB); now 256-sample gradient chunks.  Same rules.  Launch ONCE:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain4.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain4.out 2>&1 < /dev/null &
exec 9>/tmp/kagjax_speed_chain4.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
S=$R/kagjax/logs_server/D/speed
T=$R/kagjax/train
cd $R
say() { echo "[$(date -u '+%H:%M:%S') UTC] $*"; }
STOP=$(date -u -d "$(date -u -d '+1 day' +%Y-%m-%d) 02:10" +%s)
[ "$(date -u +%H)" -lt 12 ] && STOP=$(date -u -d "$(date -u +%Y-%m-%d) 02:10" +%s)
if [ "$(date -u +%s)" -ge "$STOP" ]; then say "skip exp_kl (past 02:10 UTC)"; exit 0; fi
say "queue exp_kl"
bash $R/kagjax/srv_jax.sh --gpu env PYTHONPATH=$R EXP=kl EXP_DIR=$S/exp /opt/jaxenv/bin/python $T/exp_value_kl.py > $S/exp_kl.out 2>&1
say "done  exp_kl (exit $?)"
say "CHAIN4 DONE"
