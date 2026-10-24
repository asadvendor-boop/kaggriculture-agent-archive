#!/bin/bash
# Plan step 2 (operator-approved Sep 25): DSM's scripted opening (agents/cand/DSMR0.py plays steps 0-143), then the network.
# For every fresh DSM model, as soon as its plain 464-world benchmark exists, the SAME worlds again with the DSM opening
# (tag ..._t10_dsmopen) -> paired comparison. CPU governor, one benchmark at a time.
cd /Users/asad.ali/kaggriculture/il
source live/srv_cpu.sh
ALL="c_s0 c_s1 g_s0 g_s1 x_s0 t_s0 t_s1"; DONE=""
while [ $(echo $DONE | wc -w) -lt 7 ]; do
  for k in $ALL; do
    case " $DONE " in *" $k "*) continue ;; esac
    tag=fr_${k%_s*}_dsm_s${k#*_s}
    if [ -f logs/ladder_${tag}_t10_r3_tau0.5_test.json ]; then
      PREFIX_BOT=/Users/asad.ali/kaggriculture/agents/cand/DSMR0.py PREFIX_STEPS=144 TAGX=_dsmopen bench_gov ladder ckpt/$tag.pt:10:0.5
      DONE="$DONE $k"
    elif [ -f logs/fr_${k%_s*}_s${k#*_s}.finished ] && [ ! -f ckpt/$tag.pt ]; then
      echo "[$(date -u +%H:%M) UTC] $tag has no checkpoint (training failed) - skipped"; DONE="$DONE $k"
    fi
  done
  sleep 120
done
echo "[$(date -u +%H:%M) UTC] DSM OPENING DONE"
