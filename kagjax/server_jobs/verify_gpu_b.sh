#!/bin/bash
# GPU verification batch B: replay 2048 recorded games (dataset built on the Mac with per-step C++ digests) on the
# GPU as ONE batch with each loop structure; every game must match the C++ digest at every step.
cd /Users/asad.ali/kaggriculture/kagjax
P=/opt/jaxenv/bin/python
for c in gpu seq par; do
  echo "=== replay tapes2048 $c $(date -u +%H:%M:%S)"
  bash srv_jax.sh --gpu $P gpu_verify.py replay --data logs_server/ds_tapes_mac2048.npz --cfg $c --out logs_server/replay_gpu_tapes2048_$c.json
done
echo "=== done $(date -u +%H:%M:%S)"
