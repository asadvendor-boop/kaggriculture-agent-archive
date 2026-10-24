# Sprint runbook (Sep 27-28): 8-GPU rental, 10M JAX RL, warm hour vs from-scratch hour, then the winner
All commands from the Mac unless marked [server] (run via ssh). P/H = the rental's SSH port / host.

## 0. Before renting (done on the Mac)
- trainer features + sprint.sh/preflight.sh/mem_probe.py ready (agent D report), il/live scripts synced in bootstrap.
- specialist tapes: data/tapes_spec (446 seats). Kimi tapes, if they arrive: data/tapes_kimi (same format).

## 1. Setup (target <= 25 min)
1. `bash il/live/srv_bootstrap2.sh P H`  -> copies ~200 MB, installs JAX + kaggle-env in the background, builds the C++
   engine, engine parity on 30 tapes, GPU conv test (H200: CUDA 12.4 driver -> if the conv test FAILS: stop the rental).
2. [server] `setsid nohup bash K/il/live/srv_gpu_mon.sh > /root/gpu_mon.log 2>&1 < /dev/null &`
3. [server] `cd K && bash kagjax/train/sprint.sh preflight` (~10 min; must end "PREFLIGHT OK")
4. [server] `cd K && bash kagjax/train/sprint.sh memprobe 3072,4096,5120` -> pick games/GPU with >= 15% headroom.
## 2. Hour 1: warm start on all GPUs
5. [server] `cd K && bash kagjax/train/sprint.sh A 60 <games>`   (watch the first 3 iterations: [dp] replica diff 0,
   samples/s ~ N x single GPU; if dp misbehaves -> kill, `SPRINT_MODE=seeds bash kagjax/train/sprint.sh A 60 <games>`)
6. [server] `setsid nohup bash K/il/live/srv_ckpt_eval.sh A:K/kagjax/runs/sprint_A B:K/kagjax/runs/sprint_B > K/il/logs/ckpt_eval.log 2>&1 < /dev/null &`
7. Mac: `bash il/live/watch_sprint.sh P H` (background, visible task).
## 3. Hour 2: from scratch on all GPUs (starts when A stops itself; A's state is saved)
8. [server] `cd K && bash kagjax/train/sprint.sh B 60 <games>`
## 4. Decision (~2:25 after renting)
9. [server] `bash kagjax/train/sprint.sh status` + the tester's ruler results: A vs its start / 20MW5_SE on 2400+,
   old leaders, top-10; B vs PASS / IL@0 / IL@10 trend. Switch to B only if B already beats IL@10 (strong imitation style).
   Tell the operator; if neither shows real potential -> stop the rental.
10. [server] `bash kagjax/train/sprint.sh resume <A|B> <minutes> <horizon_samples>`; last ~10%: `... resume <W> <min> <hor> win`
## 5. Uploadable checkpoints (any time a tested checkpoint clearly beats NEURAL_20MW5_SE on 2400+ AND leader tests)
11. export (Mac or server): `B64=1 python il/export.py <ckpt.pt> package/<NAME>/main.py 0.5 10` (+ `RULES="SE_GUARD=1"` if the
    SE-on test is better), tar, then package parity vs its bench JSON + official check (il/live/srv_pkgcheck.sh), turn timing.
## 6. End
12. copy results to the Mac (winner's pt/, state_*.npz, logs, eval JSONs; big -> tar + 100 MB chunks + 6 streams, see
    il/live/handoff_chunks.sh) and/or operator's Vast Cloud Copy to Drive; verify checksums.
13. stop: [server] `vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` (only after the verified copy and the
    operator's OK; the operator destroys it).
