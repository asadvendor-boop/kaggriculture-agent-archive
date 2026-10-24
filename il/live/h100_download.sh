#!/bin/bash
# Sep 27 evening: copy the 8x H100 box's results to the Mac before the operator stops/destroys it (stop 21:00-21:15 PKT).
#   h100_download.sh static   the JAX RL runs (kagjax/runs/sprint_*), their exported checkpoints, preflight logs (unchanging)
#   h100_download.sh models   finished imitation models: il/ckpt/ilsp_*.pt incl. the fresh2-pass stages (*_p.pt), not *_last.pt
#   h100_download.sh final    models again (new ones) + packages + all logs (il/logs -> il/logs/h100/, /root -> il/logs/h100/root)
#                             + the server's scripts/code (il/logs/h100/code, for the record; the Mac copies stay untouched)
#   h100_download.sh verify   every model / RL-run file / package tarball on the server exists on the Mac with the same size
# Parallel rsync streams (one per group), resumable (--partial-dir), retried.
P=23490; H=root@<SERVER_IP>; R=/Users/asad.ali/kaggriculture; cd $R
S="ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -p $P"
T=$(mktemp -d)
rs() {   # rsync with retries; exit status of rsync itself (the ssh banner lines are filtered from its messages)
  local a e st; e=$(mktemp $T/err.XXXX)
  for a in 1 2 3 4; do
    nice rsync -a --partial-dir=.rsync-partial --timeout=180 -e "$S" "$@" 2> $e; st=$?
    grep -v -E "vast.ai|Have fun|AI agents" $e; [ $st = 0 ] && return 0; sleep 5
  done
  echo "  rsync FAILED after 4 tries: $*"; return 1
}
t0=$(date +%s)
case "$1" in
  static)
    mkdir -p kagjax/runs il/ckpt
    for d in sprint_A sprint_A1 sprint_B sprint_B1; do rs $H:$R/kagjax/runs/$d kagjax/runs/ & done
    rs $H:$R/il/ckpt/A2_it4.pt $H:$R/il/ckpt/A2_it7.pt $H:$R/il/ckpt/A2_it11.pt $H:$R/il/ckpt/A_it4.pt $H:$R/il/ckpt/A_it7.pt \
       $H:$R/il/ckpt/B_it4.pt $H:$R/il/ckpt/B_it17.pt il/ckpt/ &
    rs --include '*/' --include '*.txt' --include '*.jsonl' --include '*.log' --include '*.json' --exclude '*' \
       $H:$R/kagjax/runs/preflight_A $H:$R/kagjax/runs/preflight_B kagjax/runs/ &
    wait ;;
  models|final)
    mkdir -p il/ckpt
    L=$($S $H "cd $R/il/ckpt; for m in \$(cat /root/il_bench_queue.txt); do [ -f \$m.pt ] && echo \$m.pt; [ -f \${m}_p.pt ] && echo \${m}_p.pt; done" 2>/dev/null | grep -v -E "vast.ai|Have fun|AI agents" | sort -u)
    i=0; for f in $L; do echo "$f" >> $T/dl_$((i % 4)).lst; i=$((i + 1)); done
    echo "models to copy: $(echo $L | wc -w)"
    for k in 0 1 2 3; do [ -f $T/dl_$k.lst ] && rs --files-from=$T/dl_$k.lst $H:$R/il/ckpt/ il/ckpt/ & done
    if [ "$1" = final ]; then
      mkdir -p il/logs/h100/root il/logs/h100/code package
      rs --ignore-existing $H:$R/package/ package/ &
      rs $H:$R/il/logs/ il/logs/h100/ &
      rs --include '*.log' --include '*.txt' --include '*.out' --include '*.sh' --exclude '*' $H:/root/ il/logs/h100/root/ &
      rs --exclude 'runs/' --exclude '__pycache__/' --include '*/' --include '*.py' --include '*.sh' --include '*.md' --exclude '*' \
         --prune-empty-dirs $H:$R/kagjax il/logs/h100/code/ &
      rs --include '*.py' --include '*.sh' --exclude '*' $H:$R/il/live/ il/logs/h100/code/il_live/ &
    fi
    wait ;;
  verify)   # every model / RL run file / package tarball on the server must exist on the Mac with the same size
    $S $H "cd $R; find il/ckpt -name 'ilsp_*.pt' ! -name '*_last.pt' -printf '%s %p\n'; find kagjax/runs/sprint_A kagjax/runs/sprint_A1 kagjax/runs/sprint_B kagjax/runs/sprint_B1 -type f ! -path '*.rsync-partial*' -printf '%s %p\n'; find package -name '*.tar.gz' -printf '%s %p\n'" 2>/dev/null \
      | grep -v -E "vast.ai|Have fun|AI agents" > $T/srv.lst
    [ -s $T/srv.lst ] || { echo "verify: FAILED - could not list the server's files (ssh error?)"; rm -rf $T; exit 1; }   # Sep 27: an empty list once read as '0 checked'
    bad=0; n=0
    while read -r sz p; do
      n=$((n + 1)); ms=$(stat -f %z "$p" 2>/dev/null || echo missing)
      [ "$ms" = "$sz" ] || { echo "  MISMATCH $p: server $sz, Mac $ms"; bad=$((bad + 1)); }
    done < $T/srv.lst
    echo "verify: $n server files checked, $bad missing or different on the Mac" ;;
esac
rm -rf $T
echo "[$(date -u +%H:%M:%S) UTC] h100_download $1 done in $(( $(date +%s) - t0 )) s"
