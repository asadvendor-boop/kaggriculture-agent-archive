#!/bin/bash
# Sep 27 handoff, final method: the server's two archives (/root/handoff/*.tar) split into 100 MB chunks, fetched with
# 6 parallel SSH/rsync streams (each chunk resumable via --partial-dir, retried until complete), then each archive is
# verified by streaming its chunks through sha256 against the server's SHA256SUMS. Extraction is a separate step.
P=${1:-55611}; H=${2:-<SERVER_IP>}; K=/Users/asad.ali/kaggriculture; D=$K/handoff_staging/chunks
mkdir -p $D
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $P"
LIST=$($SSH root@$H 'ls /root/handoff/chunks' 2>/dev/null | grep -E '^part[AB]\.[0-9]{3}$')
echo "[$(date -u +%H:%M:%S) UTC] $(echo "$LIST" | wc -l | tr -d ' ') chunks to fetch"; S=$(date +%s)
fetch() { for i in $(seq 1 200); do rsync -a --partial-dir=.rsync-partial --timeout=60 -e "$SSH" root@$H:/root/handoff/chunks/$1 $D/ 2>/dev/null && return 0; sleep 3; done; echo "  chunk $1 FAILED"; }
export -f fetch; export SSH H D
echo "$LIST" | xargs -P 6 -I{} bash -c 'fetch {}'
echo "[$(date -u +%H:%M:%S) UTC] fetched in $(( $(date +%s) - S )) s; $(ls $D | grep -c '^part') chunks, $(du -sh $D | cut -f1)"
cd $K/handoff_staging
for p in partA_results_tapes partB_ildata; do
  want=$(grep " $p.tar" SHA256SUMS | cut -c1-64); x=${p%%_*}
  got=$(cat $D/$x.* | shasum -a 256 | cut -c1-64)
  [ "$got" = "$want" ] && echo "  $p.tar sha256 OK" || echo "  $p.tar sha256 MISMATCH ($got vs $want)"
done
echo "CHUNKS DONE"
