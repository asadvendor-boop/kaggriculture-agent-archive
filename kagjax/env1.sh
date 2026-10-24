#!/bin/bash
# Single-core environment for running kagjax on the shared Mac (operator rule: <= 1 core for this agent).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"
export JAX_PLATFORMS=cpu
export TF_CPP_MIN_LOG_LEVEL=2
exec /Users/asad.ali/kaggriculture/.venv/bin/python "$@"
