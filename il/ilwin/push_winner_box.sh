#!/bin/bash
# From the Mac, AFTER il/live/srv_bootstrap5.sh <port> <host> (engine build, il code, v13_p3, tapes, rulers):
# copies what the winner fine-tune needs on top: these scripts (+ the live packages' rule sources), tapes_st2 (not in
# bootstrap5) + fresh copies of the live tape dirs, the rehearsal subset (1.39 GB: 3,000 fresh2 + 1,537 dsm_ft2 seats + x6).
# Usage: bash push_winner_box.sh <port> <host>
PORT=$1; HOST=$2; K=/Users/asad.ali/kaggriculture; S=$(cd "$(dirname "$0")" && pwd)
RS() { for a in 1 2 3 4 5 6; do rsync -az --partial-dir=.rsync-partial --timeout=120 -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT" "$@" && return 0; sleep 5; done; echo "RSYNC FAILED: $*"; return 1; }
T0=$(date +%s)
ssh -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST "mkdir -p $K/il/ilwin $K/il/data/reh_fresh2 $K/il/data/reh_dsm2 $K/il/data/x6 $K/data"
RS --exclude='dryrun' --exclude='__pycache__' --exclude='reh_lists' $S/ root@$HOST:$K/il/ilwin/
for d in tapes_st2 tapes_soh tapes_n20mw8 tapes_npse tapes_nps tapes_n20mw5; do RS $K/data/$d root@$HOST:$K/data/; done
RS --include='rows_*.json' --exclude='*' $K/data/ root@$HOST:$K/data/          # dates / ratings per game (--since)
RS $K/il/ckpt/ilsp_v13_p3.pt root@$HOST:$K/il/ckpt/                           # (bootstrap5 copies it too; no-op then)
RS $K/il/data/team_vocab.json $K/il/data/rating_w_fresh2.json root@$HOST:$K/il/data/
echo "[$(date -u +%H:%M:%S) UTC] code + tapes done ($(( $(date +%s) - T0 )) s); rehearsal subset (1.39 GB) next"
RS -L --files-from=$S/reh_lists/reh_fresh2.txt $K/il/data/fresh2/ root@$HOST:$K/il/data/reh_fresh2/
RS -L --files-from=$S/reh_lists/reh_dsm2.txt $K/il/data/dsm_ft2/ root@$HOST:$K/il/data/reh_dsm2/
RS --files-from=$S/reh_lists/reh_x6.txt $K/il/data/x6/ root@$HOST:$K/il/data/x6/
ssh -p $PORT root@$HOST "ls $K/il/data/reh_fresh2 | wc -l; ls $K/il/data/reh_dsm2 | wc -l; ls $K/il/data/x6 | wc -l; du -sh $K/il/data/reh_fresh2 $K/il/data/reh_dsm2 $K/il/data/x6"
echo "[$(date -u +%H:%M:%S) UTC] push done ($(( $(date +%s) - T0 )) s) - expect 3000 / 1537 / >= 4537"
