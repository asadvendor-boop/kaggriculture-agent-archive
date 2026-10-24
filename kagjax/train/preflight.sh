#!/bin/bash
# New machine, before anything else (~5 min, blocking): is everything the A / B runs need there and working?
#   bash kagjax/train/preflight.sh
# GPUs + JAX/CUDA, checkpoints, tape dirs (tapes.py check: selection + exact C++ replay), the C++ engine importable from
# $PY (the trainer builds the TOP / SPEC banks with it at start-up), the option tests on CPU, a 20-world TOP / SPEC
# pinning re-check (the full 100-world gate passed on the Mac, Sep 27), and a 2-iteration single-GPU smoke of each recipe.
R=/Users/asad.ali/kaggriculture
PY=${PY:-/opt/jaxenv/bin/python}
export PYTHONPATH=$R:$PYTHONPATH TF_CPP_MIN_LOG_LEVEL=2
ok=1
say() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
fail() { echo "  FAIL: $*"; ok=0; }
say "GPUs: $(nvidia-smi -L | wc -l)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c
$PY -c "import jax; print('  jax', jax.__version__, jax.devices())" || fail "JAX / CUDA"
for f in rf_w_dsm_s0 rf_w_s0; do [ -f $R/il/ckpt/$f.pt ] || fail "missing il/ckpt/$f.pt"; done
for d in tapes_top12 tapes_spec; do [ -f $R/data/$d/index.json ] || fail "missing data/$d/index.json"; done
JAX_PLATFORMS=cpu $PY -c "import sys; sys.path.insert(0, '$R/tools'); import harness, topruler; print('  C++ engine', harness.ENGINE_DIR)" \
  || fail "C++ engine (tools/harness + engine/v2) not importable from $PY: build engine/v2 for this Python"
say "tape dirs"
JAX_PLATFORMS=cpu $PY $R/kagjax/train/tapes.py check --dirs $R/data/tapes_top12 --spec $R/data/tapes_spec,$R/data/tapes_spec_more \
  | grep -E "^\[" || fail "tapes.py check"
say "option tests (CPU, mock)"
mkdir -p /tmp/kagjax_preflight
JAX_PLATFORMS=cpu TEST_DIR=/tmp/kagjax_preflight $PY $R/kagjax/train/test_ab_options.py all > /tmp/kagjax_preflight/tests.out 2>&1
grep -E "PASS|FAIL" /tmp/kagjax_preflight/tests.out | cut -c1-160; grep -q "ALL PASS" /tmp/kagjax_preflight/tests.out || fail "test_ab_options"
say "TOP / SPEC pinning re-check, 10 + 10 worlds (CPU)"
JAX_PLATFORMS=cpu RECHECK_OUT=/tmp/kagjax_preflight/tape_recheck.json $PY $R/kagjax/train/recheck_tape_pin.py 10 10 | grep -E "learner\]|GATE" \
  || fail "recheck_tape_pin"
grep -q '"PASS": true' /tmp/kagjax_preflight/tape_recheck.json 2>/dev/null || fail "TOP / SPEC gate"
for r in A B; do
  say "recipe $r: 2 iterations on GPU 0 (B=256, no evaluation)"
  out=$(CUDA_VISIBLE_DEVICES=0 DEVICES=1 MODE=dp bash $R/kagjax/train/run_gpu.sh preflight_$r RECIPE=$r B=256 K=4 ITERS=2 EVAL_EVERY=0 \
        EVAL_EVERY_MIN=0 CKPT_EVERY=1000 SNAP_EVERY=1000 PT_EVERY_SAMPLES=0)
  pid=$(echo "$out" | grep -o 'PID [0-9]*' | awk '{print $2}')
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  grep -E "^\[tapes\]|^it [0-9]|STOP" $R/kagjax/runs/preflight_$r/log.txt | cut -c1-200
  grep -q "^it 2 " $R/kagjax/runs/preflight_$r/log.txt || { fail "recipe $r smoke"; tail -5 $R/kagjax/runs/preflight_$r/stdout.log; }
done
[ $ok = 1 ] && say "PREFLIGHT OK -> bash kagjax/train/sprint.sh memprobe; then bash kagjax/train/sprint.sh A 60" || say "PREFLIGHT FAILED (see above)"
