#!/bin/bash
cd /Users/asad.ali/kaggriculture/kagjax
echo "=== start $(date -u +%H:%M:%S)"
bash srv_jax.sh --gpu /opt/jaxenv/bin/python gpu_verify_all.py
echo "=== end $(date -u +%H:%M:%S)"
