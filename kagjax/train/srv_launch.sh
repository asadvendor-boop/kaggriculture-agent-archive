#!/bin/bash
# Detached launch on the server (the SSH session returns at once; the host drops long sessions):
#   ssh ... 'bash /Users/asad.ali/kaggriculture/kagjax/train/srv_launch.sh <log> [VAR=value ...] <cmd ...>'
# Runs <cmd> with the VAR=value pairs exported, through the shared GPU wrapper when <cmd> starts with --gpu
# (kagjax/srv_jax.sh: cores 56-63, nice 10, one JAX GPU job at a time, 20% of the A100's memory).
LOG=${1:?log}; shift
while [[ "$1" == *=* ]]; do export "$1"; shift; done
export PYTHONPATH=/Users/asad.ali/kaggriculture:$PYTHONPATH
mkdir -p "$(dirname "$LOG")"
cd /Users/asad.ali/kaggriculture
setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/srv_jax.sh "$@" > "$LOG" 2>&1 < /dev/null &
echo "PID $! -> $LOG"
