#!/bin/bash
# Self-play PPO (kagjax/train/ppo.py) on a GPU server.  Launches detached (setsid nohup) and returns at once.
#
# Usage:  bash kagjax/train/run_gpu.sh <run_name> [VAR=value ...]
#   MODE=dp     (default) one process, data parallel over all visible GPUs (games sharded, params replicated,
#               gradients averaged; B is per GPU, so 4 GPUs = 4x the games per iteration)
#   MODE=seeds  one independent process per GPU (CUDA_VISIBLE_DEVICES=i, SEED=i, RUN_DIR=<run>/s<i>); pick / soup later
#   WRAP="bash /Users/asad.ali/kaggriculture/kagjax/srv_jax.sh --gpu"   on the SHARED A100 (lock + 20% memory cap)
#   PY=/opt/jaxenv/bin/python (JAX with CUDA), any ppo.py variable (INIT, B, K, SEG, MB, EPOCHS, LR, BETA, OPPS, ...)
# Outputs: kagjax/runs/<run_name>/ (log.txt, eval.jsonl, pt/it*.pt, best.pt, state_*.npz, snaps/), stdout.log
# Monitor: tail -f kagjax/runs/<run_name>/log.txt ; stop: kill the PID printed below (and nothing else).
# Export a checkpoint for Kaggle:  python il/export.py kagjax/runs/<run>/best.pt package/<NAME>/main.py 0.5 10
#   (RULES="SE_GUARD=1" python il/export.py ... bakes the SE guard into the package, as for the torch pipeline)
set -e
R=/Users/asad.ali/kaggriculture
NAME=${1:?usage: run_gpu.sh <run_name> [VAR=value ...]}; shift
for kv in "$@"; do export "$kv"; done
PY=${PY:-/opt/jaxenv/bin/python}
MODE=${MODE:-dp}
RUN=${RUN_ROOT:-$R/kagjax/runs}/$NAME
mkdir -p "$RUN"
export PYTHONPATH=$R:$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
# production defaults (override on the command line)
export KAGJAX_BACKEND=${KAGJAX_BACKEND:-real}
# RECIPE (operator plan Sep 27, 10M only; design.md "Runs A and B" / "Sprint"): run defaults layered on PRESET=fast.
#   A = warm start from the weighted 10M (rf_w_dsm_s0, team 10), win bonus 0.5, KL anchor loosened over the horizon;
#   B = from scratch (random 10M weights), no KL anchor, entropy bonus annealed over the horizon, T=1 sampling, value
#       head trained with the trunk, win bonus ramped in over the first 300M samples.
#   Both: the LRs follow the global minibatch (LR_SCALE=sqrt: x sqrt(GPUs x MB / 4096)), bf16 opponent forwards,
#   evaluation every 12 min (fp32, 64 games per GPU per opponent), a .pt every 100M samples (ring of 6 + one per 500M
#   + the final), state every 2 iterations, SE guard off, midnight / final-fix / homing rules on.
RECIPE=${RECIPE:-}
if [ "$RECIPE" = A ]; then
  MODEL=${MODEL:-10m}
  export WINB=${WINB:-0.5}
  export BETA=${BETA:-0.05} BETA_END=${BETA_END:-0.005} BETA_SCHEDULE=${BETA_SCHEDULE:-exp} BETA_DECAY_FRAC=${BETA_DECAY_FRAC:-0.9}
  export LR=${LR:-2e-5} VLR=${VLR:-4e-4}                   # per 4,096-sample minibatch (x sqrt(8) = 5.7e-5 / 1.1e-3 on 8 GPUs)
  export OPPS=${OPPS:-LATEST=3,SELF=3,IL=1.5,TOP=1,SPEC=1.5}   # x PFSP: ~60% self / past, ~40% clones + leaders + specialists
  export EVAL_OPPS=${EVAL_OPPS:-SNAP0,IL@10}
elif [ "$RECIPE" = B ]; then
  MODEL=${MODEL:-10m}
  export INIT=${INIT:-random}                          # architecture from IL (rf_w_s0: 256x8, 48 teams)
  export BETA=${BETA:-0} ENT_COEF=${ENT_COEF:-0.01} ENT_END=${ENT_END:-0.001} T=${T:-1} KSHARP=${KSHARP:-1}
  export LR=${LR:-1e-4} VLR=${VLR:-4e-4}                   # per 4,096-sample minibatch (x sqrt(8) = 2.8e-4 / 1.1e-3 on 8 GPUs)
  export VALUE_TARGET=${VALUE_TARGET:-togo} VALUE_GRAD=${VALUE_GRAD:-0.5} VWARM=${VWARM:-0}
  export WINB=${WINB:-0.5} WINB_RAMP_SAMPLES=${WINB_RAMP_SAMPLES:-3e8}
  export OPPS=${OPPS:-LATEST=6,SELF=4,IL=1,TOP=0.5,SPEC=0.5,PASS=0.5} IL_TEAMS=${IL_TEAMS:-0:1}   # x PFSP: ~70% self / past
  export EVAL_OPPS=${EVAL_OPPS:-PASS,IL@0,IL@10}
elif [ -n "$RECIPE" ]; then
  echo "unknown RECIPE=$RECIPE (A or B)"; exit 2
fi
if [ -n "$RECIPE" ]; then
  export TAPE_DIRS=${TAPE_DIRS:-$R/data/tapes_top12} SPEC_DIRS=${SPEC_DIRS:-$R/data/tapes_spec,$R/data/tapes_spec_more}
  export TAPE_MAX=${TAPE_MAX:-4096} SPEC_MAX=${SPEC_MAX:-4096}    # all training seats (Sep 27: 2,423 leaders, 322 spec.)
  export B=${B:-4096} K=${K:-8} SEG=${SEG:-720} LR_SCALE=${LR_SCALE:-sqrt} LR_REF_MB=${LR_REF_MB:-4096} OPP_PRECISION=${OPP_PRECISION:-bf16}
  export HORIZON_SAMPLES=${HORIZON_SAMPLES:-1e9}
  export EVAL_EVERY=${EVAL_EVERY:-0} EVAL_EVERY_MIN=${EVAL_EVERY_MIN:-12} EVAL_N=${EVAL_N:-64}
  export SNAP_EVERY=${SNAP_EVERY:-2} SELF_MAX=${SELF_MAX:-16} CKPT_EVERY=${CKPT_EVERY:-2}
  export PT_EVERY_SAMPLES=${PT_EVERY_SAMPLES:-1e8} PT_KEEP=${PT_KEEP:-6} PT_MILESTONE_SAMPLES=${PT_MILESTONE_SAMPLES:-5e8}
  export PT_EVERY_MIN=${PT_EVERY_MIN:-0} PT_EVERY=${PT_EVERY:-0} ITERS=${ITERS:-100000}
  export SE_GUARD=${SE_GUARD:-0} MIDNIGHT=${MIDNIGHT:-1} FINALFIX=${FINALFIX:-1} HOMING=${HOMING:-1}
fi
[ -n "${XLA_EXTRA:-}" ] && export XLA_FLAGS="${XLA_FLAGS:-} $XLA_EXTRA"      # optional XLA flags (A/B them: sprint.sh)
# MODEL=20m (default) / 10m: the operator's rating-weighted warm starts (Sep 26) and their multi-style parents
MODEL=${MODEL:-20m}
if [ "$MODEL" = 10m ]; then D_INIT=$R/il/ckpt/rf_w_dsm_s0.pt; D_IL=$R/il/ckpt/rf_w_s0.pt
else D_INIT=$R/il/ckpt/g20_w_dsm_s0.pt; D_IL=$R/il/ckpt/g20_w_s0.pt; fi
export INIT=${INIT:-$D_INIT}                          # warm start (team 10 = DSM style); KL anchor REF defaults to INIT
export IL=${IL-$D_IL}                                 # style-clone opponents: the same-size multi-style model before the
                                                      # DSM fine-tune (must have INIT's architecture), played as IL_TEAMS
export TEAM=${TEAM:-10} T=${T:-0.2} KSHARP=${KSHARP:-10} TAU=${TAU:-0.5}
# PRESET=fast (default, Sep 27; kagjax/train/design.md "Update speed"): 1 epoch of 4,096-sample minibatches (MICRO
#   accumulation), LR by the square-root rule, the KL anchor on a 25% subsample (unbiased), the policy's own value head
#   (stop-gradient) predicting the final margin, TF32 forward values with bf16 gradients: ~4-5x the gate's update speed.
# PRESET=gate: the gate-5 settings (2 epochs x MB=256, a separate 20M value network, TF32 everywhere).
PRESET=${PRESET:-fast}
if [ "$PRESET" = gate ]; then
  export EPOCHS=${EPOCHS:-2} MB=${MB:-256} LR=${LR:-5e-6} VLR=${VLR:-1e-4} VWARM=${VWARM:-1}
else
  export EPOCHS=${EPOCHS:-1} MB=${MB:-4096} MICRO=${MICRO:-1024} LR=${LR:-2e-5} VLR=${VLR:-4e-4} VWARM=${VWARM:-1}
  export REF_FRAC=${REF_FRAC:-0.25} VALUE_INIT=${VALUE_INIT:-head} VALUE_TARGET=${VALUE_TARGET:-final}
  export UPDATE_PRECISION=${UPDATE_PRECISION:-bf16grad}
fi
export BETA=${BETA:-0.05}
export B=${B:-1024} K=${K:-4} SEG=${SEG:-720}   # keep B/K >= 256 (one opponent forward per block)
# league: current learner (LATEST), past selves (SELF), team-conditioned clones of the imitation model (IL as IL_TEAMS:
# DSM 10, Vadim 5, DECEM 3, UMG 4, Boey 0) and recorded leader games replayed in their exact worlds (TOP, tapes.py:
# data/tapes_top12, benchmark episodes excluded, ep % 10 == 0 held out for the TOP evaluation)
# (TOP / SPEC: tape worlds pinned like the C++ reference; gate recheck_tape_pin.py passed Sep 27, 100/100 worlds)
export OPPS=${OPPS:-LATEST=2,SELF=2,IL=4} IL_TEAMS=${IL_TEAMS:-10:3,5:2,3:2,4:2,0:2} TAPE_MAX=${TAPE_MAX:-1024}
export EVAL_EVERY=${EVAL_EVERY:-5} EVAL_N=${EVAL_N:-256} EVAL_OPPS=${EVAL_OPPS:-SNAP0,IL@10,IL@5,IL@0}   # SNAP0 = the start policy
export SNAP_EVERY=${SNAP_EVERY:-5} SELF_MAX=${SELF_MAX:-8} CKPT_EVERY=${CKPT_EVERY:-5} PT_EVERY=${PT_EVERY:-0} PT_EVERY_MIN=${PT_EVERY_MIN:-90}
export ITERS=${ITERS:-1000}
if [ "$MODE" = seeds ]; then                          # one run per GPU of CUDA_VISIBLE_DEVICES (default: all GPUs)
  GPUS=${CUDA_VISIBLE_DEVICES:-$(seq 0 $(( $(nvidia-smi -L | wc -l | tr -d ' ') - 1 )) | paste -sd, -)}
  for i in ${GPUS//,/ }; do
    mkdir -p "$RUN/s$i"
    CUDA_VISIBLE_DEVICES=$i SEED=$(( ${SEED:-0} + i )) RUN_DIR="$RUN/s$i" DEVICES=1 setsid nohup $WRAP $PY $R/kagjax/train/ppo.py \
      > "$RUN/s$i/stdout.log" 2>&1 < /dev/null &
    echo "seed $(( ${SEED:-0} + i )) on GPU $i: PID $! -> $RUN/s$i/log.txt"
  done
else
  RUN_DIR="$RUN" setsid nohup $WRAP $PY $R/kagjax/train/ppo.py > "$RUN/stdout.log" 2>&1 < /dev/null &
  echo "PID $! -> $RUN/log.txt"
fi
