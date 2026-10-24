#!/bin/bash
# Mac -> server sync for the Sep 26 iteration (code changed today + our live tapes/rows), then the three chains, each
# detached as its own statement (setsid nohup ... &): RL3 first (fixed 12 workers), then the refresh (downloads, builds),
# then the rule tests (the CPU governor gives benchmarks what is left).   Usage: sync_day3.sh [ssh port] [host]
PORT=${1:-13692}; HOST=${2:-<SERVER_IP>}; S="ssh -o ConnectTimeout=15 -p $PORT root@$HOST"
K=/Users/asad.ali/kaggriculture
set -e
rsync -az -e "ssh -p $PORT" $K/il/features.py $K/il/live_rollout.py $K/il/ppo_live.py $K/il/train6.py root@$HOST:$K/il/
rsync -az -e "ssh -p $PORT" --include='*.py' --include='*.sh' --exclude='*' $K/il/live/ root@$HOST:$K/il/live/
rsync -az -e "ssh -p $PORT" $K/data/rows_56553797.json $K/data/rows_56553911.json $K/data/rows_56559691.json $K/data/rows_56571775.json root@$HOST:$K/data/
rsync -az -e "ssh -p $PORT" $K/data/tapes_n10md $K/data/tapes_n10mr root@$HOST:$K/data/
$S "mkdir -p $K/data/tapes_nlive && cp -n $K/data/tapes_n10md/*.json.gz $K/data/tapes_n10mr/*.json.gz $K/data/tapes_nlive/ 2>/dev/null; ls $K/data/tapes_nlive | wc -l"
set +e
$S "cd $K/il && rm -f logs/fr_pending && bash -n live/srv_rl3.sh && bash -n live/srv_refresh.sh && bash -n live/srv_rules.sh && /venv/main/bin/python -c 'import features, live_rollout, ppo_live' && echo CODE OK"
