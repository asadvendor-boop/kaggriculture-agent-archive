#!/bin/bash
# Morning handoff (operator, Sep 26 ~21:30 UTC): copy all important server data to the Mac before the Vast instance is
# STOPPED (never destroyed - the operator destroys it later). Pull-only, and it never overwrites the Mac's master code:
# checkpoints / training data / il/live / data / packages are copied with --ignore-existing (only files the Mac lacks);
# il/logs go to il/logs/server2/; the server's kagjax/ (without compile caches), kagjax_ext/, il/live and /root/Downloads
# are also mirrored in full to server_mirror/. Resumable (--partial) with retries. Ends with a verification: a dry run
# of every copy must find nothing left, and the sha256 of the newest checkpoints must match on both sides.
# Usage: morning_handoff.sh [port] [host]
P=${1:-55611}; H=${2:-<SERVER_IP>}; K=/Users/asad.ali/kaggriculture; S=$K/server_mirror
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $P"; R=root@$H:$K
mkdir -p $S $K/il/logs/server2
FAIL=0
# partial transfers go to .rsync-partial (resumed on the retry); the destination file appears only when complete, so
# --ignore-existing can never freeze a half-copied file (Sep 27 fix: --partial + --ignore-existing did exactly that)
RS() { for a in 1 2 3 4 5 6 7 8 9 10 11 12; do rsync -a --partial-dir=.rsync-partial --timeout=120 -e "$SSH" "$@" 2>/dev/null && return 0; sleep 15; done; echo "  rsync FAILED: $*"; FAIL=1; return 1; }
LEFT() { rsync -a --partial-dir=.rsync-partial --dry-run --itemize-changes --timeout=120 -e "$SSH" "$@" 2>/dev/null | grep -c '^>f'; }
COPIES=(
  "--ignore-existing|$R/il/ckpt/|$K/il/ckpt/"
  "--ignore-existing|$R/il/data/|$K/il/data/"
  "--ignore-existing --exclude=__pycache__|$R/il/live/|$K/il/live/"
  "|$R/il/logs/|$K/il/logs/server2/"
  "--ignore-existing|$R/data/|$K/data/"
  "--ignore-existing|$R/package/|$K/package/"
  "--exclude=.jax_cache --exclude=cov_build --exclude=__pycache__|$R/kagjax/|$S/kagjax/"
  "--exclude=__pycache__|$R/kagjax_ext/|$S/kagjax_ext/"
  "--exclude=__pycache__|$R/il/live/|$S/il_live/"
  "|root@$H:/root/Downloads/|$S/root_Downloads/"
)
echo "[$(date -u +%H:%M) UTC] handoff sync starts; Mac free: $(df -h / | tail -1 | awk '{print $4}')"
for c in "${COPIES[@]}"; do IFS='|' read -r f src dst <<< "$c"; mkdir -p "$dst"; echo "  copy $src -> $dst"; RS $f "$src" "$dst"; done
echo "[$(date -u +%H:%M) UTC] verifying"
for c in "${COPIES[@]}"; do IFS='|' read -r f src dst <<< "$c"; n=$(LEFT $f "$src" "$dst"); echo "  left to copy: $n  ($src)"; [ "$n" = 0 ] || FAIL=1; done
NEW=$($SSH root@$H "cd $K && ls -t il/ckpt/*.pt 2>/dev/null | head -12; ls -t kagjax/logs_server/D/overnight/*.pt 2>/dev/null | head -4" 2>/dev/null | grep '\.pt$')
for f in $NEW; do
  r=$($SSH root@$H "sha256sum $K/$f" 2>/dev/null | cut -c1-16)
  case $f in kagjax/*) l=$(shasum -a 256 $S/$f 2>/dev/null | cut -c1-16);; *) l=$(shasum -a 256 $K/$f 2>/dev/null | cut -c1-16);; esac
  [ -n "$r" ] && [ "$r" = "$l" ] && echo "  sha OK   $f" || { echo "  sha DIFF $f (server $r, mac $l)"; FAIL=1; }
done
echo "[$(date -u +%H:%M) UTC] Mac free now: $(df -h / | tail -1 | awk '{print $4}')"
[ $FAIL = 0 ] && echo "HANDOFF SYNC VERIFIED" || echo "HANDOFF SYNC INCOMPLETE"
