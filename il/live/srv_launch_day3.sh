#!/bin/bash
# New server, Sep 26: waits for the operator's data bundle (Google Drive -> /root/kag_bundle.tar), unpacks it without
# overwriting anything already here (newer code, the leaders' newest tapes), runs the setup checks, then starts the three
# chains detached: RL3 (league RL), the rule tests, and - once the leader prefetch has finished - the data refresh.
K=/Users/asad.ali/kaggriculture; L=$K/il/logs/launch_day3.log
log() { echo "[$(date -u +%H:%M) UTC] $*" | tee -a $L; }
until [ "$(stat -c %s /root/kag_bundle.tar 2>/dev/null)" = "769473536" ]; do sleep 20; done
log "bundle complete: $(sha256sum /root/kag_bundle.tar | cut -c1-16) (Mac: 67f909a4947f4433)"
tar xf /root/kag_bundle.tar -C / --skip-old-files 2>&1 | tail -3 | tee -a $L
log "bundle unpacked: ckpt $(ls $K/il/ckpt | tr '\n' ' '); union $(ls $K/il/data/union 2>/dev/null | wc -l); x6 $(ls $K/il/data/x6 2>/dev/null | wc -l); tapes_kd days $(ls $K/data/tapes_kd | tr '\n' ' ')"
bash $K/il/live/srv_setup_bundle.sh 2>&1 | tee -a $L
grep -q "engine parity: 30 exact, 0 different" $L && grep -q "imports OK" $L || { log "SETUP CHECK FAILED - chains not started"; exit 1; }
cd $K/il
NW_RL=${NW_RL:-20} setsid nohup bash live/srv_rl3.sh 7 > logs/rl3_chain.log 2>&1 < /dev/null &
log "RL3 chain started (pid $!)"
sleep 90
setsid nohup bash live/srv_rules.sh > logs/rules.log 2>&1 < /dev/null &
log "rules chain started (pid $!)"
until grep -q "PREFETCH DONE" logs/prefetch.log 2>/dev/null; do sleep 30; done
cd $K
NB=${NB:-10} setsid nohup bash il/live/srv_refresh.sh > il/logs/refresh_chain.log 2>&1 < /dev/null &
log "refresh chain started (pid $!)"
