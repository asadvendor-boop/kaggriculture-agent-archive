#!/bin/bash
# GPU verification batch C: replay 512 bot+perturbation fuzz games (built on the Mac with per-step C++ digests) on the
# GPU as ONE batch with each loop structure; every game must match the C++ digest at every step.
cd /Users/asad.ali/kaggriculture/kagjax
P=/opt/jaxenv/bin/python
for c in gpu seq par; do
  echo "=== replay fuzz512 $c $(date -u +%H:%M:%S)"
  bash srv_jax.sh --gpu $P gpu_verify.py replay --data logs_server/ds_fuzz_mac512.npz --cfg $c --out logs_server/replay_gpu_fuzz512_$c.json
done
echo "=== done $(date -u +%H:%M:%S)"
