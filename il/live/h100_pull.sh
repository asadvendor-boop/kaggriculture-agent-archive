#!/bin/bash
# Sep 27 evening: the 8x H100 box -> Mac link is capped at ~0.07 MB/s PER CONNECTION (server upload 51 MB/s, Mac
# download 5.5 MB/s, 12 parallel ssh streams 0.78 MB/s) -> bundles are tarred on the server, split into 16 MB parts and
# pulled over many parallel ssh connections (no multiplexing: every part has its own TCP connection), checked by size,
# then streamed through tar on the Mac (no whole-tar copy on the nearly full disk) with a SHA-256 check of the stream.
#   h100_pull.sh pack <name> <path> ...   (server) tar the paths (relative to the repo) -> /root/xfer/<name>.part_*
#   h100_pull.sh pull <name> [streams]    (Mac) fetch the parts (default 24 in parallel), verify, extract into the repo
#                                         with tar -k (never overwrites a Mac file), delete the parts
P=23490; H=root@<SERVER_IP>; R=/Users/asad.ali/kaggriculture
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ControlMaster=no -o ControlPath=none -p $P"
case "$1" in
  pack)
    n=$2; shift 2
    $S $H "mkdir -p /root/xfer && cd $R && rm -f /root/xfer/$n.* && tar cf - $* | split -b 16M -a 3 - /root/xfer/$n.part_ && cat /root/xfer/$n.part_* | sha256sum | cut -c1-64 > /root/xfer/$n.sha256 && ls -l /root/xfer/$n.part_* | awk '{print \$5, \$9}' > /root/xfer/$n.list && echo \"$n: \$(wc -l < /root/xfer/$n.list) parts, \$(awk '{s+=\$1} END {print s}' /root/xfer/$n.list) bytes, sha \$(cut -c1-16 /root/xfer/$n.sha256)\"" 2>&1 \
      | grep -v -E "vast.ai|Have fun|AI agents" ;;
  pull)
    n=$2; J=${3:-24}; D=$R/.xfer_$n; mkdir -p $D; cd $D
    $S $H "cat /root/xfer/$n.list; echo SHA \$(cat /root/xfer/$n.sha256)" 2>/dev/null | grep -v -E "vast.ai|Have fun|AI agents" > $D/list.txt
    sha=$(awk '$1 == "SHA" {print $2}' list.txt); grep -v '^SHA' list.txt > parts.txt
    echo "[$(date -u +%H:%M:%S) UTC] $n: $(wc -l < parts.txt | tr -d ' ') parts, $(awk '{s+=$1} END {printf "%.0f MB", s/1e6}' parts.txt), $J streams"
    fetch() {   # fetch <size> <remote path>: retried until the local copy has exactly the server's size
      local sz=$1 f=$2 b; b=$(basename $2)
      for a in 1 2 3 4 5 6 7 8; do
        [ -f $b ] && [ "$(stat -f %z $b)" = "$sz" ] && return 0
        ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ControlMaster=no -o ControlPath=none -p 23490 root@<SERVER_IP> "cat $f" > $b.tmp 2>/dev/null \
          && [ "$(stat -f %z $b.tmp)" = "$sz" ] && mv $b.tmp $b && return 0
        sleep $((a * 3))
      done
      echo "  FAILED $b"; return 1
    }
    export -f fetch
    t0=$(date +%s)
    awk '{print $1, $2}' parts.txt | xargs -P $J -n 2 bash -c 'fetch "$0" "$1"'
    bad=0; while read -r sz f; do b=$(basename $f); [ -f $b ] && [ "$(stat -f %z $b)" = "$sz" ] || { echo "  missing/short $b"; bad=$((bad + 1)); }; done < parts.txt
    dt=$(( $(date +%s) - t0 )); echo "[$(date -u +%H:%M:%S) UTC] $n: fetched in $dt s ($(awk -v t=$dt '{s+=$1} END {printf "%.2f", s/1e6/(t>0?t:1)}' parts.txt) MB/s), $bad parts missing"
    [ $bad = 0 ] || exit 1
    got=$(cat $(awk '{print $2}' parts.txt | xargs -n1 basename) | tee >(cd $R && tar xkf - 2>&1 | grep -v -E "Already exists|File exists" | head -5) | shasum -a 256 | cut -c1-64)
    sleep 2
    if [ "$got" = "$sha" ]; then echo "[$(date -u +%H:%M:%S) UTC] $n: SHA-256 OK ($(echo $sha | cut -c1-16)), extracted into $R (existing files kept)"; cd $R && rm -rf $D
    else echo "[$(date -u +%H:%M:%S) UTC] $n: SHA-256 MISMATCH (server $sha, Mac $got) - parts kept in $D"; exit 1; fi ;;
esac
