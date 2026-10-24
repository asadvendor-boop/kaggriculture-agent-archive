# CPU governor for the server chains (operator, Sep 25: "keep atleast 2 cores free for system"). The container has ~30.7
# cores -> 28 for our jobs. Counted: trainers ~1.1 core each (data loading), every pool worker 1, the fresh data build
# while it runs (15 until FRESH READY, then 8 until FRESH7 READY), and fresh trainers not started yet (training has
# priority over benchmarks). Benchmarks run one at a time (lock) with 6..24 workers from what is left.
# Sep 26: numpy-run packages spawned one BLAS thread per core each (server load 634) -> every job that sources this
# file runs 1 math thread per process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ -z "$CPU_BUDGET" ]; then           # Sep 26: from the container's CPU quota (new machine), >= 2 cores kept free
  CORES=""
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q per < /sys/fs/cgroup/cpu.max; [ "$q" != max ] && CORES=$((q / per)); fi
  [ -z "$CORES" ] && CORES=$(nproc)
CPU_BUDGET=$((CORES - 2))
  [ $CPU_BUDGET -gt 50 ] && CPU_BUDGET=50  # operator, Sep 26 (61-core server): up to ~50 cores for all our jobs
fi
GOV_LOGS=/Users/asad.ali/kaggriculture/il/logs
cpu_used() {
  local t s b r=0 p=0
  t=$(pgrep -fc "python train6.py"); s=$(pgrep -fc "multiprocessing.spawn")
  b=$(pgrep -fc "il/build_dataset3.py|il/live/srv_build_x6.py")          # Sep 26: data-refresh builders, 1 core each
  if ! grep -q "FRESH READY" $GOV_LOGS/fresh_data.log 2>/dev/null; then r=15
  elif ! grep -q "FRESH7 READY" $GOV_LOGS/fresh_data.log 2>/dev/null; then r=8; fi
  [ -f $GOV_LOGS/fr_pending ] && p=$(cat $GOV_LOGS/fr_pending)
  local a=0; [ -f /tmp/agent_cores ] && a=$(cat /tmp/agent_cores)   # Sep 26: cores reserved for the JAX sub-agents
  echo $(( (11 * (t + p) + 9) / 10 + s + b + r + a ))
}
cpu_free() { echo $(( CPU_BUDGET - $(cpu_used) )); }
bench_gov() {   # $1 = ladder | top:<team>; then checkpoints (ladder: ckpt:team:tau specs)
  local kind=$1 n; shift
  exec 9>/tmp/srv_bench.lock; flock 9
  while [ "$(cpu_free)" -lt 6 ]; do sleep 60; done
  n=$(cpu_free); [ $n -gt ${BENCH_CAP:-40} ] && n=${BENCH_CAP:-40}   # (24 on the old 31-core server; BENCH_CAP on big boxes)
  echo "[$(date -u +%H:%M) UTC] $kind benchmark with $n workers: $*"
  if [ "$kind" = ladder ]; then
    NW=$n LOCKSTEP=6 MIDNIGHT=1 FINALFIX=1 HOMING=1 /venv/main/bin/python live/bench_multi.py "$@" 2>&1 | grep --line-buffered -v Warning
  else
    local day=fresh; [ "${kind%%:*}" = top2 ] && day=fresh2          # top2 = today's top-10 benchmark (Sep 26)
    DAY=$day N=300 NW=$n TEAM=${kind#*:} TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 /venv/main/bin/python live/top_bench.py "$@" 2>&1 | grep --line-buffered -v Warning
  fi
  flock -u 9; exec 9>&-
}
