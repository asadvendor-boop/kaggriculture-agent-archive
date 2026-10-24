#!/bin/bash
# Package a trained checkpoint as a Kaggle submission: numpy-only main.py (export.py) -> package/<NAME>/main.py + tar.gz,
# then verify: loader + C++ engine parity (torch agent vs exported main.py on one leader world) and timing.
# Usage: il/package.sh <ckpt> <NAME> <tau> <team>
set -e
cd /Users/asad.ali/kaggriculture
CK=$1; NAME=$2; TAU=$3; TEAM=$4
mkdir -p package/$NAME
nice -n 15 .venv/bin/python il/export.py $CK package/$NAME/main.py $TAU $TEAM
(cd package/$NAME && COPYFILE_DISABLE=1 tar czf ../$NAME.tar.gz main.py)
ls -la package/$NAME.tar.gz
shasum -a 256 package/$NAME/main.py | cut -c1-16
nice -n 15 .venv/bin/python - <<EOF
import sys, json, time; sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch; torch.set_num_threads(1)
from harness import mk
import topruler
from agent import factory
p, seat = json.load(open("il/eval_worlds.json"))[0]
t0 = time.time(); a = topruler.run_tape(p, 1 - seat, mk("/Users/asad.ali/kaggriculture/package/$NAME/main.py")); t1 = time.time()
b = topruler.run_tape(p, 1 - seat, factory("$CK", mk_tau=$TAU, team=$TEAM))
print("exported main.py bank", a[0], "vs torch agent", b[0], "->", "IDENTICAL" if abs(a[0] - b[0]) < .5 else "DIFFERENT", "| %.1fs per game" % (t1 - t0))
EOF
