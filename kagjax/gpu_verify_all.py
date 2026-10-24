"""All GPU verification runs in ONE process (one acquisition of the shared GPU lock):
  replay 2,048 recorded games and 512 fuzz games (C++ digests) with CFG_GPU / CFG_SEQ / CFG_PAR,
  batch independence (B=1 / K / 4096) with CFG_GPU (K=7) and CFG_SEQ / CFG_PAR (K=3: B=1 runs are slow with dynamic loops).
"""
import argparse
import time
import gpu_verify as G

L = "logs_server"
t0 = time.time()
for data, tag in ((f"{L}/ds_tapes_mac2048.npz", "tapes2048"), (f"{L}/ds_fuzz_mac512.npz", "fuzz512")):
    for c in ("gpu", "seq", "par"):
        print(f"=== replay {tag} {c} t+{time.time() - t0:.0f}s", flush=True)
        G.replay(argparse.Namespace(data=data, cfg=c, n=0, out=f"{L}/replay_gpu_{tag}_{c}.json"))
for c, k, sizes in (("gpu", 7, "1,7,4096"), ("seq", 3, "1,3,4096"), ("par", 3, "1,3,4096")):
    print(f"=== batchindep {c} t+{time.time() - t0:.0f}s", flush=True)
    G.batchindep(argparse.Namespace(data=f"{L}/ds_tapes_mac64.npz", cfg=c, k=k, sizes=sizes,
                                    out=f"{L}/batchindep_gpu_{c}.json"))
print(f"=== all done t+{time.time() - t0:.0f}s", flush=True)
