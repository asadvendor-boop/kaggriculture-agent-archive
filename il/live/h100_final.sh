#!/bin/bash
# Sep 27 evening, the LAST copy before the operator's 16:00-16:15 UTC stop of the 8x H100 box:
#   every imitation model (il/ckpt/ilsp_*.pt incl. *_p.pt, not *_last.pt) and package tarball not yet on the Mac (name +
#   size), all logs (server il/logs -> il/logs/h100/, /root/*.log|txt|out|sh -> il/logs/h100/root/) and the server's
#   scripts (il/live -> il/logs/h100/code/il_live/, kagjax code -> il/logs/h100/code/kagjax/; the Mac's own code untouched),
#   as one bundle through h100_pull.sh (16 MB parts, parallel connections, SHA-256, tar -k), then h100_download.sh verify.
#   h100_final.sh models [streams]   only the new models / packages (bundle 'models2'; logs are copied once, at the end,
#                                    because tar -k never replaces a file that already exists on the Mac)
#   h100_final.sh all [streams]      new models / packages + logs + scripts (bundle 'final'), then verify   (default 32)
P=23490; H=root@<SERVER_IP>; R=/Users/asad.ali/kaggriculture; cd $R
S="ssh -o ConnectTimeout=20 -p $P"
T=$(mktemp -d)
{ for f in il/ckpt/ilsp_*.pt package/*.tar.gz; do [ -f "$f" ] && echo "$f $(stat -f %z "$f")"; done; } | sort > $T/have.txt
$S $H "cd $R; for f in il/ckpt/ilsp_*.pt package/*.tar.gz; do case \$f in *_last.pt) continue;; esac; echo \"\$f \$(stat -c %s \$f)\"; done" 2>/dev/null \
  | grep -E "^(il/ckpt|package)/" | sort > $T/srv.txt
NEW=$(comm -13 $T/have.txt $T/srv.txt | awk '{print $1}')
CODE=$($S $H "cd $R; find il/live -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \); find kagjax -path kagjax/runs -prune -o -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) -print" 2>/dev/null | grep -E "^(il/live|kagjax)/")
ROOTF=$($S $H "cd / && ls root/*.log root/*.txt root/*.out root/*.sh 2>/dev/null" 2>/dev/null | grep "^root/")
MODE=${1:-all}; N=final; [ "$MODE" = models ] && { N=models2; CODE=""; ROOTF=""; }
echo "[$(date -u +%H:%M:%S) UTC] $N bundle: $(echo $NEW | wc -w | tr -d ' ') new models/packages, il/logs, $(echo $ROOTF | wc -w | tr -d ' ') /root files, $(echo $CODE | wc -w | tr -d ' ') code files"
echo "$NEW" | sed 's/^/  new: /'
if [ "$MODE" = models ]; then
  [ -n "$NEW" ] && bash il/live/h100_pull.sh pack models2 $NEW && bash il/live/h100_pull.sh pull models2 ${2:-32}
else
  bash il/live/h100_pull.sh pack final "--transform='s,^il/logs/,il/logs/h100/,'" "--transform='s,^root/,il/logs/h100/root/,'" \
    "--transform='s,^il/live/,il/logs/h100/code/il_live/,'" "--transform='s,^kagjax/,il/logs/h100/code/kagjax/,'" \
    $NEW il/logs $CODE -C / $ROOTF
  bash il/live/h100_pull.sh pull final ${2:-32}
  bash il/live/h100_download.sh verify
fi
rm -rf $T
