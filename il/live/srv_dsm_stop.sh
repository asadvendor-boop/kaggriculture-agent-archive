#!/bin/bash
# Sep 28: stop the DSM plan screen (il/live/srv_dsm_rules.sh, PID given) once its l8 result is written: m4 (melon wave)
# lost 7 leader-test wins, so m4l8 / plan / m6 (all with the melon wave) are not worth the cores. Kills by explicit PID:
# the script first, then its children and their pool workers.
P=$1; L=/root/a100_bench.log
until grep -q "== DSM l8" $L; do sleep 20; done
sleep 5
kids=$(ps -o pid= --ppid $P); gk=""; for c in $kids; do gk="$gk $(ps -o pid= --ppid $c)"; done
kill $P 2>/dev/null; kill $kids $gk 2>/dev/null
echo "  (DSM screen stopped after l8 at $(date -u +%H:%M) UTC: m4 lost 7 leader-test wins, so m4l8 / plan / m6 were cancelled)" >> $L
