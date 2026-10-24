"""Extract the embedded features.py / np_runtime.py / features6.py sources from a live package tar.gz (streamed in
memory, str.find only - no regex over the 94 MB main.py). Usage: extract_pkg_src.py <pkg.tar.gz> <out_prefix>"""
import sys, tarfile, ast, hashlib
pkg, out = sys.argv[1], sys.argv[2]
with tarfile.open(pkg, "r:gz") as tf:
    m = tf.getmember("main.py")
    src = tf.extractfile(m).read().decode("utf-8")
print(pkg, "main.py", len(src), "chars sha256", hashlib.sha256(src.encode()).hexdigest()[:16])
def grab(tag):
    end = src.find(f', "{tag}", "exec")')
    if end < 0:
        end = src.find(f", '{tag}', 'exec')")
    if end < 0:
        return None
    start = src.rfind("exec(compile(", 0, end)
    lit = src[start + len("exec(compile("):end]
    return ast.literal_eval(lit)
for tag, name in (("features.py", "features"), ("np_runtime.py", "np_runtime"), ("features6.py", "features6")):
    s = grab(tag)
    if s is None:
        print("  missing", tag); continue
    open(f"{out}_{name}.py", "w").write(s)
    print(f"  {tag}: {len(s)} chars -> {out}_{name}.py")
# agent wrapper tail (small)
tail = src[src.rfind("def _fallback"):]
open(f"{out}_tail.py", "w").write(tail)
i = src.find("_AGENT = _R.NumpyAgent("); print("  ", src[i:i+120].splitlines()[0])
i = src.find("_NET = _R.NumpyNet("); print("  ", src[i:i+160].splitlines()[0])
