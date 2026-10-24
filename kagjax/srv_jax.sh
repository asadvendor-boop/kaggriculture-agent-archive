#!/bin/bash
# Sub-agent jobs on the A100 server (Sep 26, operator: "why don't we take all these three on cloud server too?").
# Usage (on the server):  bash /Users/asad.ali/kaggriculture/kagjax/srv_jax.sh [--gpu] <command ...>
# Limits, so the live RL run (RL4) and the 20M training on the same machine are never hurt:
#  - every job is pinned to cores 56-63 (the 8 cores reserved for the JAX track; the CPU governor of the other chains
#    subtracts them via /tmp/agent_cores), nice 10, one math thread per process;
#  - JAX never preallocates GPU memory; --gpu jobs wait for a shared lock (ONE JAX GPU job at a time over all agents)
#    and are capped at 20% of the 40 GB A100 (the RL run and the training use ~28 GB);
#  - jobs without --gpu see no GPU at all (JAX_PLATFORMS=cpu).
# JAX with CUDA lives in its own venv (/opt/jaxenv, sees torch/numpy of /venv/main); never pip install into /venv/main.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export PATH=/opt/jaxenv/bin:$PATH
if [ "$1" = --gpu ]; then
  shift; exec flock /tmp/kagjax_gpu.lock taskset -c 56-63 nice -n 10 "$@"
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
exec taskset -c 56-63 nice -n 10 "$@"
