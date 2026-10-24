#!/bin/bash
# Sep 29 night, unattended: build upload-READY packages (never uploaded; the operator decides) for S (now) and S2 (after its 464),
# with the EXACT ALLG rule layer. Checks: embedded rule code identical to the live ALLG package's, 12-world exact parity vs the harness
# 464 rows, official kaggle_environments loader plays both seats. Log il/logs/pkg_night.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; KE=/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/ke_venv/bin/python
LOG=$K/il/logs/pkg_night.log; SUM=$K/il/logs/night29_summary.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG $SUM; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="SE_GUARD=1,STRADDLE=2,SHEEP_MIRROR=1,SHEEP_MGAP=4,SHEEP_HOLD=1,SHEEP_KEEP=1,LAND_GUARD=2,SHEEP_LAST_DAY=15,SHEEP_FLOOR=150,GOOSE_EGG=2,GOOSE_YARN=2,GOOSE_GAP=3,GOOSE_BATCH=6,GOOSE_FROM_DAY=10,GOOSE_HOUR=16,GOOSE_LAST_DAY=13,COW_CARE=2,STRAW2=4,STRAW2_P=120,STRAW2_FROM=11,COW2=2,COW2_FLOOR=170,COW2_LAST_DAY=12,STRADDLE_PREBUY=1"
mk() { c=$1; N=$2; J=$3
  mkdir -p $K/package/$N
  B64=1 RULES="$R" RULES_STR="SELL_ORDER=held" nice -n 15 $PY export.py ckpt/$c.pt $K/package/$N/main.py 0.5 10 > logs/mkpkg_$N.log 2>&1 || { say "PKG $N export FAILED: $(tail -2 logs/mkpkg_$N.log)"; return 1; }
  $PY - $K/package/$N <<'PYEOF' 2>&1 | tee -a $LOG $SUM
import sys, ast, tarfile, hashlib, os
d = sys.argv[1]; name = os.path.basename(d); src = open(f"{d}/main.py").read()
def grab(s, tag):
    e = s.find(f', "{tag}", "exec")'); e = e if e >= 0 else s.find(f", '{tag}', 'exec')")
    st = s.rfind("exec(compile(", 0, e); return ast.literal_eval(s[st + len("exec(compile("):e])
ref = "/Users/asad.ali/kaggriculture/il/ilwin/pkg_src/NEURAL_20MW8_ST2_ALLG"
same = [grab(src, f"{t}.py") == open(f"{ref}_{t}.py").read() for t in ("features", "np_runtime", "features6")]
tp = f"{d}.tar.gz"
with tarfile.open(tp, "w:gz") as tf:
    ti = tf.gettarinfo(f"{d}/main.py", arcname="main.py"); ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
    with open(f"{d}/main.py", "rb") as fh: tf.addfile(ti, fh)
mem = [m.name for m in tarfile.open(tp).getmembers()]
print(f"  PKG {name}: rule code identical to live ALLG (features/np_runtime/features6) = {same} | tar {os.path.getsize(tp):,} B sha256 {hashlib.sha256(open(tp,'rb').read()).hexdigest()[:16]} | members {mem}")
PYEOF
  nice -n 15 $PY live/pkg_parity.py $K/package/$N/main.py $J 12 > logs/pkgparity_$N.log 2>&1; say "  PKG $N parity: $(tail -1 logs/pkgparity_$N.log | cut -c1-150)"
  nice -n 15 $KE - $K/package/$N/main.py <<'PYEOF' 2>&1 | tail -2 | tee -a $LOG $SUM
import sys
from kaggle_environments import make
p = sys.argv[1]; env = make("kaggriculture", configuration={"seed": 7}, debug=False); env.run([p, p])
st = env.steps[-1]; print(f"  OFFICIAL kaggle_environments seed 7: statuses {[s.status for s in st]} rewards {[s.reward for s in st]} steps {len(env.steps)}")
PYEOF
}
say "PKG night: S now"
mk ilsp_v13_win_S NEURAL_20MW8_ALLG_S logs/ladder_ilsp_v13_win_S_t10_r3_tau0.5_soh_st2_wg_test.json
J2=logs/ladder_ilsp_v13_win_S2_t10_r3_tau0.5_soh_st2_wg_test.json
until [ -f $J2 ] || grep -qE "NIGHT CHAIN DONE|TRAIN S2 FAILED" logs/win29_chain.log; do sleep 60; done
[ -f $J2 ] && [ -f ckpt/ilsp_v13_win_S2.pt ] && { say "PKG night: S2"; mk ilsp_v13_win_S2 NEURAL_20MW8_ALLG_S2 $J2; }
say "PKG night DONE"
