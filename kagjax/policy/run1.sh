#!/bin/bash
# Run a Python command on ONE CPU core (shared Mac: 4 agents, 4 cores total).
# XLA CPU ignores --xla_cpu_multi_thread_eigen in jax 0.11; PJRT_NPROC/NPROC=1 is what limits its thread pool
# (measured: 7.6 cores -> 0.97 cores on a 2048^2 matmul loop).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PJRT_NPROC=1 NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS_EXTRA}"
export JAX_PLATFORMS=cpu
export PYTHONPATH=/Users/asad.ali/kaggriculture:${PYTHONPATH}
cd /Users/asad.ali/kaggriculture
exec nice -n 5 /Users/asad.ali/kaggriculture/.venv/bin/python "$@"
