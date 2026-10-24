#!/bin/bash
# Run a Python script on ONE CPU core (shared Mac: several agents, <= 1 core for this one).  Execution is
# single-threaded with PJRT_NPROC / NPROC = 1 (as kagjax/policy/run1.sh); XLA's compiler still spawns short bursts of
# threads, so the process also runs at background QoS (efficiency cores, lowest priority) with a cheaper XLA
# optimisation level (fine for the tiny CPU tests; never used on the GPU server).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PJRT_NPROC=1 NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS_EXTRA}"
export JAX_PLATFORMS=cpu
export PYTHONPATH=/Users/asad.ali/kaggriculture:${PYTHONPATH}
cd /Users/asad.ali/kaggriculture
exec taskpolicy -c background nice -n 10 /Users/asad.ali/kaggriculture/.venv/bin/python "$@"
