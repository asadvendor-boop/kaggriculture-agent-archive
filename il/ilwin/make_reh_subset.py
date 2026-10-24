#!/usr/bin/env python3
"""Rehearsal subset of v13_p3's own training data for the upload to the GPU box (the Mac uploads slowly):
  reh_fresh2 = N random fresh2 seats that are NOT DSM fine-tune seats (+ every fresh2 validation seat, ep % 37 == 0)
  reh_dsm2   = every dsm_ft2 seat (v13_p3's last stage: DSM, team 10)
Writes <out>/reh_fresh2.txt, reh_dsm2.txt (npz basenames), reh_x6.txt (x6 basenames) and prints sizes + rsync commands
(rsync -L dereferences the fresh2 / dsm_ft2 symlinks, so the box gets real files in il/data/reh_fresh2 and il/data/reh_dsm2).
Usage: make_reh_subset.py <out_dir> [N=3000] [seed=29]"""
import sys, os, glob, random
K = os.environ.get("KROOT", "/Users/asad.ali/kaggriculture")
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 29
os.makedirs(out, exist_ok=True)
x6 = f"{K}/il/data/x6"
has6 = lambda b: os.path.exists(f"{x6}/{b[:-4]}.npy")
dsm = sorted(os.path.basename(f) for f in glob.glob(f"{K}/il/data/dsm_ft2/*.npz") if has6(os.path.basename(f)))
fr = sorted(os.path.basename(f) for f in glob.glob(f"{K}/il/data/fresh2/*.npz") if has6(os.path.basename(f)))
dset = set(dsm)
pool = [b for b in fr if b not in dset]
val = [b for b in pool if int(b.split("_")[0]) % 37 == 0]
rest = [b for b in pool if int(b.split("_")[0]) % 37 != 0]
random.Random(seed).shuffle(rest)
pick = sorted(val + rest[:max(0, N - len(val))])
open(f"{out}/reh_fresh2.txt", "w").write("\n".join(pick) + "\n")
open(f"{out}/reh_dsm2.txt", "w").write("\n".join(dsm) + "\n")
xs = sorted({b[:-4] + ".npy" for b in pick + dsm})
open(f"{out}/reh_x6.txt", "w").write("\n".join(xs) + "\n")
sz = lambda d, bs: sum(os.path.getsize(os.path.realpath(f"{d}/{b}")) for b in bs)
a = sz(f"{K}/il/data/fresh2", pick); b = sz(f"{K}/il/data/dsm_ft2", dsm); c = sz(x6, xs)
print(f"reh_fresh2: {len(pick)} seats ({len(val)} validation) {a/1e9:.2f} GB | reh_dsm2: {len(dsm)} seats {b/1e9:.2f} GB | "
      f"x6: {len(xs)} files {c/1e9:.2f} GB | total {(a+b+c)/1e9:.2f} GB (fresh2 pool {len(pool)}, dsm_ft2 {len(dsm)})")
print("upload (from the Mac; PORT/HOST of the box):")
print(f"  rsync -aL --files-from={out}/reh_fresh2.txt -e 'ssh -p $PORT' {K}/il/data/fresh2/ root@$HOST:{K}/il/data/reh_fresh2/")
print(f"  rsync -aL --files-from={out}/reh_dsm2.txt  -e 'ssh -p $PORT' {K}/il/data/dsm_ft2/ root@$HOST:{K}/il/data/reh_dsm2/")
print(f"  rsync -a  --files-from={out}/reh_x6.txt    -e 'ssh -p $PORT' {K}/il/data/x6/ root@$HOST:{K}/il/data/x6/")
