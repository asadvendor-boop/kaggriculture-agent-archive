#!/bin/bash
# Trimmed GPU jobs, each takes the shared lock separately (short holds).  Launch detached.
R=/Users/asad.ali/kaggriculture; W="bash $R/kagjax/srv_jax.sh --gpu /opt/jaxenv/bin/python -m kagjax.policy.bench_gpu"
export PYTHONPATH=$R
cd $R
BENCH_BS=256,1024 BENCH_REPS=10 BENCH_G=256 $W speed $R/il/ckpt/g20_u_dsm_s0.pt
BENCH_SKIP_FWD=1 BENCH_REPS=10 BENCH_G=256 $W speed $R/il/ckpt/fr_t_dsm_s0.pt
N_PARITY=128 $W parity $R/il/ckpt/fr_t_dsm_s0.pt $R/il/ckpt/g20_u_dsm_s0.pt
echo GPU_CHAIN_DONE
