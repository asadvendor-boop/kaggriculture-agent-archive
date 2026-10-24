#!/bin/bash
# Sep 28 evening: follow the rented box from the Mac (reconnects; prints only new lines): /root/tonight.log (all lines),
# errors in /root/tn.out and the variant logs, and one line when srv_tonight.sh is no longer running.
# Usage: mac_watch_tonight.sh <port> <host>
PORT=$1; HOST=$2; N=${3:-0}; E=${4:-0}; ENDED=0
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -o BatchMode=yes -o LogLevel=ERROR -p $PORT root@$HOST"
while true; do
  OUT=$($S "tail -n +$((N + 1)) /root/tonight.log 2>/dev/null" 2>/dev/null | grep -v "vast\|Have fun\|AI agents")
  if [ -n "$OUT" ]; then printf '%s\n' "$OUT"; N=$((N + $(printf '%s\n' "$OUT" | wc -l))); fi
  ERR=$($S "cat /root/tn.out /root/tn_fast.out /Users/asad.ali/kaggriculture/il/logs/tn_*.log 2>/dev/null | grep -E 'Traceback|Error|Killed|MemoryError' | grep -v -i warn | wc -l" 2>/dev/null | grep -E '^[0-9]+$')
  if [ -n "$ERR" ] && [ "$ERR" -gt "$E" ]; then echo "BOX ERROR lines: $ERR (was $E) - check /root/tn.out and il/logs/tn_*.log"; E=$ERR; fi
  RUN=$($S "pgrep -fc '[s]rv_(gpu_chain|h2h_st|gpu_fresh3|tonight|wave)'" 2>/dev/null | grep -E '^[0-9]+$')
  if [ "$RUN" = 0 ] && [ $ENDED = 0 ]; then echo "ALL BOX TEST CHAINS ENDED"; ENDED=1; fi
  sleep 90
done
