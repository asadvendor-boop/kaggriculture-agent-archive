#!/bin/bash
# Gate item 5: learning sanity on the real engine (warm start g20_w_dsm_s0; opponents: IL clones g20_w_s0 as DSM /
# Vadim / DECEM / UMG / Boey, past selves, the current learner), production PPO settings at a small batch that fits
# the shared A100's 8 GB cap; greedy evaluation vs the frozen start (SNAP0) and the IL clone as DSM on 128 fixed worlds
# before and after.  NaN / KL guards stop the run.  Launch through srv_launch.sh (--gpu).
R=/Users/asad.ali/kaggriculture
D=$R/kagjax/logs_server/D
export KAGJAX_BACKEND=real INIT=$R/il/ckpt/g20_w_dsm_s0.pt IL=$R/il/ckpt/g20_w_s0.pt
export B=${B:-128} K=4 SEG=720 MB=256 EPOCHS=2 LR=5e-6 VLR=1e-4 BETA=0.05 VWARM=1 T=0.2 KSHARP=10 TAU=0.5 TEAM=10
export OPPS=LATEST=2,SELF=2,IL=4 IL_TEAMS=10:3,5:2,3:2,4:2,0:2 SNAP_EVERY=1 SELF_MAX=4
export EVAL_EVERY=${EVAL_EVERY:-3} EVAL_N=128 EVAL_OPPS=SNAP0,IL@10 CKPT_EVERY=1 PT_EVERY=3 ITERS=${ITERS:-3} KL_MAX=1.0
export RUN_DIR=$D/gate5
exec /opt/jaxenv/bin/python $R/kagjax/train/ppo.py
