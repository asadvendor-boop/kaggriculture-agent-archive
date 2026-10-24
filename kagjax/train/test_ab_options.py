"""Tests of the Sep 27 options for the 10M A / B runs (Mac CPU; mock backend for the trainer runs).

unit      entropy bonus terms of policy_loss vs numpy (units over legal ops at temperature T, quantities, Bernoulli fires);
          the bonus's gradient raises the entropy; KL off (refp None) = no KL terms; BETA / WINB schedules.
mock A    warm-start-like run: BETA 0.05 -> 0.005 (exp over 3 iterations), WINB 0.5 ramped over 2 iterations, PASS
          opponents in OPPS and EVAL_OPPS, value head, PT_EVERY_SAMPLES; checks the logged beta / winb / exports.
mock B    from-scratch-like run: BETA 0 (no reference terms: ref_samples 0, KL 0), ENT_COEF, VALUE_GRAD=0.5, T=1,
          KSHARP=1, resume keeps the sample counter.
dp        mock B with 2 host devices (shard_map data parallel): replica differences logged and 0, same stats keys.
dpresume  2 iterations on 1 device, RESUME on 2 devices: sample counter and BETA schedule continue.
usage: python kagjax/train/test_ab_options.py [unit|mockA|mockB|dp|dpresume|all]      (JAX_PLATFORMS=cpu)
"""
import os, sys, json, glob, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = "/Users/asad.ali/kaggriculture"
OUT = os.environ.get("TEST_DIR", "/tmp/kagjax_test_ab")
PY = sys.executable


def unit():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import ppo as P
    import sample as S
    rng = np.random.RandomState(0)
    N, U, O, Q, H = 6, 16, 44, 12, 21
    out = {"op": jnp.asarray(rng.randn(N, U, O).astype(np.float32)), "qty": jnp.asarray(rng.randn(N, U, Q).astype(np.float32)),
           "mk": jnp.asarray(rng.randn(N, 249).astype(np.float32) * 2)}
    umask = jnp.asarray(rng.rand(N, U, O) < 0.3).at[:, :, 0].set(True)
    uact = jnp.asarray(np.where(np.arange(U)[None] < 5, 0, -1).repeat(N, 0), jnp.int8)
    qact = jnp.asarray(np.where(np.arange(U)[None] < 3, 1, -1).repeat(N, 0), jnp.int8)
    mfire = jnp.asarray(rng.rand(N, H) < 0.3)
    T, KS, tau = 0.7, 3.0, jnp.full(H, 0.5)
    cur = S.logp_terms(out, umask, uact, qact, mfire, T, KS, tau)
    d = dict(valid=jnp.ones(N, bool), umask=umask)
    old = {k: cur[k] for k in ("lpu", "lpq", "lpm")}
    a = jnp.asarray(rng.randn(N).astype(np.float32))
    _, st = P.policy_loss(cur, d, a, old, None, 0.2, 0.05, ent_coef=0.0)
    # numpy reference entropies
    z = np.where(np.asarray(umask), np.asarray(out["op"]) / T, -np.inf)
    pz = np.exp(z - z.max(-1, keepdims=True)); pz /= pz.sum(-1, keepdims=True)
    hu = -(np.where(pz > 0, pz * np.log(np.where(pz > 0, pz, 1)), 0)).sum(-1)[:, :5].mean()
    qz = np.asarray(out["qty"]) / T; pq = np.exp(qz - qz.max(-1, keepdims=True)); pq /= pq.sum(-1, keepdims=True)
    hq = -(pq * np.log(pq)).sum(-1)[:, :3].mean()
    pf = np.asarray(cur["pf"], np.float64); hm = -(pf * np.log(pf) + (1 - pf) * np.log(1 - pf)).mean()
    res = dict(ent_u=(float(st["ent"]), hu), ent_q=(float(st["ent_q"]), hq), ent_m=(float(st["ent_m"]), hm), klu_off=float(st["klu"]))
    ok = all(abs(x - y) < 2e-4 * max(1, abs(y)) for x, y in (res["ent_u"], res["ent_q"], res["ent_m"])) and res["klu_off"] == 0.0

    # the bonus's gradient raises the entropy (loss = -c H: a small step along -grad increases H)
    def total_ent(o):
        c = S.logp_terms(o, umask, uact, qact, mfire, T, KS, tau)
        l, s_ = P.policy_loss(c, d, a * 0, {k: jax.lax.stop_gradient(c[k]) for k in ("lpu", "lpq", "lpm")}, None, 0.2, 0.0, ent_coef=1.0)
        return l, s_["ent"] + 0.5 * s_["ent_q"] + s_["ent_m"]
    g = jax.grad(lambda o: total_ent(o)[0])(out)
    h0 = float(total_ent(out)[1])
    h1 = float(total_ent(jax.tree_util.tree_map(lambda x, y: x - 0.05 * y, out, g))[1])
    res["entropy_before_after_step"] = (h0, h1)
    ok &= h1 > h0

    class C:
        beta, beta_end, beta_decay_iters, beta_decay_samples, beta_schedule, winb, winb_ramp_iters = 0.05, 0.005, 100, 0.0, "exp", 0.5, 10
    b = [P.beta_schedule(C, it, 0) for it in (1, 51, 101, 500)]
    C.beta_schedule = "linear"; bl = [P.beta_schedule(C, it, 0) for it in (1, 51, 101)]
    C.beta_decay_samples = 1e9; C.beta_schedule = "exp"; bs = [P.beta_schedule(C, 7, s) for s in (0, 5e8, 1e9, 3e9)]
    C.beta_end = 0.0; C.beta_schedule = "exp"; b0 = [P.beta_schedule(C, 1, s) for s in (0, 5e8, 1e9)]      # exp to 0 -> linear
    C.beta_end = -1.0; bc = P.beta_schedule(C, 999, 1e12)
    w = [P.winb_schedule(C, it) for it in (1, 5, 10, 20)]
    res.update(beta_exp_iters=b, beta_linear=bl, beta_exp_samples=bs, beta_to_zero=b0, beta_const=bc, winb_ramp=w)
    ok &= abs(b[0] - 0.05) < 1e-12 and abs(b[1] - 0.05 * 0.1 ** 0.5) < 1e-9 and abs(b[2] - 0.005) < 1e-12 and abs(b[3] - 0.005) < 1e-12
    ok &= abs(bl[1] - 0.0275) < 1e-12 and abs(bs[1] - 0.05 * 0.1 ** 0.5) < 1e-9 and b0 == [0.05, 0.025, 0.0] and bc == 0.05
    ok &= w == [0.05, 0.25, 0.5, 0.5]
    print("UNIT", "PASS" if ok else "FAIL", json.dumps(res, default=float), flush=True)
    return ok


def run(name, env, devices=1, fresh=True):
    d = os.path.join(OUT, name)
    if fresh:
        shutil.rmtree(d, ignore_errors=True)
    e = dict(os.environ, KAGJAX_BACKEND="mock", INIT="tiny:8x1", IL="", B="8", K="4", SEG="48", EPISODE_SLOTS="48", MB="16",
             EVAL_N="8", CKPT_EVERY="0", PT_EVERY="0", SEED="3", RUN_DIR=d, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
             PYTHONPATH=R)
    e.update(env)
    e["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" + (
        f" --xla_force_host_platform_device_count={devices}" if devices > 1 else "")
    if devices > 1:
        e["JAX_NUM_CPU_DEVICES"] = str(devices)          # (JAX 0.11 ignores the XLA flag on the Mac)
    r = subprocess.run([PY, os.path.join(HERE, "ppo.py")], env=e, capture_output=True, text=True)
    log = open(os.path.join(d, "log.txt")).read() if os.path.exists(os.path.join(d, "log.txt")) else ""
    st = [json.loads(l) for l in open(os.path.join(d, "stats.jsonl"))] if os.path.exists(os.path.join(d, "stats.jsonl")) else []
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode, log, st, d


def mock_a():
    rc, log, st, d = run("mockA", dict(ITERS="4", BETA="0.05", BETA_END="0.005", BETA_DECAY_ITERS="3", BETA_SCHEDULE="exp", WINB="0.5",
                                       WINB_RAMP_ITERS="2", OPPS="LATEST=2,SELF=2,PASS=2", EVAL_EVERY="2", EVAL_OPPS="SNAP0,PASS",
                                       VALUE_INIT="head", VALUE_TARGET="final", EPOCHS="1", MICRO="8", REF_FRAC="0.5",
                                       PT_EVERY_SAMPLES="700", CKPT_EVERY="2"))
    betas = [round(s["beta"], 6) for s in st]; winbs = [s["winb"] for s in st]
    exp_b = [round(0.05 * 0.1 ** (x / 3), 6) for x in (0, 1, 2, 3)]
    pts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "pt", "*.pt")))
    passw = "PASS" in log and "vs PASS" in log
    ok = rc == 0 and betas == exp_b and winbs == [0.25, 0.5, 0.5, 0.5] and all(s["ref_samples"] > 0 for s in st) and passw
    ok &= pts == ["it2.pt", "it4.pt"]            # 384 samples / iteration: 700 -> after it 2 (768), 1400 -> it 4 (1536)
    print("MOCK A", "PASS" if ok else "FAIL", dict(rc=rc, betas=betas, expected=exp_b, winbs=winbs, pts=pts, pass_in_log=passw,
          ref_samples=[s["ref_samples"] for s in st]), flush=True)
    return ok


def mock_b(devices=1, name="mockB"):
    env = dict(ITERS="3", BETA="0", ENT_COEF="0.01", VALUE_INIT="head", VALUE_TARGET="togo", VALUE_GRAD="0.5", T="1", KSHARP="1",
               OPPS="LATEST=2,SELF=2,PASS=1", EVAL_EVERY="3", EVAL_OPPS="PASS,SNAP0", EPOCHS="1", MICRO="8", CKPT_EVERY="3",
               WINB="0.3", WINB_RAMP_ITERS="3")
    if devices > 1:
        env.update(DEVICES=str(devices))
    rc, log, st, d = run(name, env, devices)
    ok = rc == 0 and len(st) == 3 and all(s["ref_samples"] == 0 and s["klu"] == 0 and s["beta"] == 0 for s in st)
    ok &= all(s["ent"] > 0 and s["ent_q"] >= 0 and s["ent_m"] > 0 for s in st) and all(s["finite"] == 1 for s in st)
    res = dict(rc=rc, ent=[round(s["ent"], 3) for s in st], ent_q=[round(s["ent_q"], 3) for s in st], ent_m=[round(s["ent_m"], 3) for s in st],
               loss=[round(s["loss"], 4) for s in st])
    if devices > 1:
        diffs = [s.get("dp_replica_diff") for s in st]
        res["dp_replica_diff"] = diffs
        ok &= all(x is not None and x == 0.0 for x in diffs) and f"x{devices}" in log
    else:                                       # resume: the sample counter continues (stats "samples" 1152 -> 1536)
        e2 = dict(env, ITERS="4", RESUME=os.path.join(d, "state_3.npz"))
        rc2, log2, st2, _ = run(name, e2, fresh=False)
        st_all = [json.loads(l) for l in open(os.path.join(d, "stats.jsonl"))]
        res["resume_rc"] = rc2; res["samples"] = [s["samples"] for s in st_all]
        ok &= rc2 == 0 and res["samples"] == [384, 768, 1152, 1536]
    print(("DP" if devices > 1 else "MOCK B"), "PASS" if ok else "FAIL", json.dumps(res), flush=True)
    return ok


def dp_resume():
    """Moving a run to more GPUs: 2 iterations on 1 device, then RESUME on 2 devices (sample counter continues)."""
    env = dict(ITERS="2", BETA="0.05", BETA_END="0.005", BETA_DECAY_SAMPLES="3000", VALUE_INIT="head", EPOCHS="1", MICRO="8",
               OPPS="LATEST=2,SELF=2,PASS=1", EVAL_EVERY="0", CKPT_EVERY="2", SNAP_EVERY="1")
    rc1, _, _, d = run("dpResume", env)
    rc2, log, st, _ = run("dpResume", dict(env, ITERS="3", RESUME=os.path.join(d, "state_2.npz"), DEVICES="2"), devices=2, fresh=False)
    smp = [s["samples"] for s in st]
    betas = [round(s["beta"], 6) for s in st]
    exp_b3 = round(0.05 * 0.1 ** (768 / 3000), 6)            # beta of iteration 3 = f(samples before it: 2 x 384)
    ok = rc1 == 0 and rc2 == 0 and smp == [384, 768, 1536] and betas[-1] == exp_b3 and "[resume]" in log and "x2" in log
    print("DP RESUME", "PASS" if ok else "FAIL", json.dumps(dict(rc=(rc1, rc2), samples=smp, betas=betas, expected_beta3=exp_b3)), flush=True)
    return ok


def sprint():
    """Sprint features: MAX_MINUTES stop (final eval, state, pt, FINAL.json), MARGIN_W=0 + WINB=1 (win-only returns),
    horizon-aware ENT / BETA schedules continued across a RESUME with a longer horizon, VWARM_RESUME, LR_SCALE=sqrt,
    the pt ring (PT_KEEP + milestones + final), games.jsonl."""
    res, ok = {}, True
    # (1) wall-clock stop: ITERS huge, MAX_MINUTES 0.4 (24 s; start-up counts)
    rc, log, st, d = run("sprintStop", dict(ITERS="100000", MAX_MINUTES="0.4", EVAL_EVERY_MIN="0.1", EVAL_OPPS="SNAP0", VALUE_INIT="head",
                                            EPOCHS="1", MICRO="8", CKPT_EVERY="1000", PT_EVERY_SAMPLES="1e9"))
    fin = json.load(open(os.path.join(d, "FINAL.json"))) if os.path.exists(os.path.join(d, "FINAL.json")) else {}
    res["stop"] = dict(rc=rc, iters=len(st), final=fin, stop_line="[stop] MAX_MINUTES" in log,
                       files=all(os.path.exists(x) for x in (fin.get("state", "-"), fin.get("pt", "-"))))
    ok &= rc == 0 and res["stop"]["stop_line"] and res["stop"]["files"] and fin.get("minutes", 99) <= 0.5 and len(st) >= 1
    # (2) win-only returns: rew_sum x B = sum of sign(final margin) (integers), and horizon-aware ENT / BETA
    env = dict(ITERS="4", MARGIN_W="0", WINB="1", ENT_COEF="0.01", ENT_END="0", ENT_DECAY_FRAC="1", HORIZON_SAMPLES="1536",
               BETA="0.05", BETA_END="0.005", BETA_DECAY_FRAC="1", VALUE_INIT="head", VALUE_TARGET="final", EPOCHS="1", MICRO="8",
               EVAL_EVERY="0", CKPT_EVERY="2", SNAP_EVERY="1", PT_EVERY_SAMPLES="384", PT_KEEP="1", PT_MILESTONE_SAMPLES="1000")
    rc, log, st, d = run("sprintWin", env)
    sums = [s_["rew_sum"] * 8 for s_ in st]
    ents = [round(s_["ent_coef"], 6) for s_ in st]
    betas = [round(s_["beta"], 6) for s_ in st]
    exp_b = [round(0.05 * 0.1 ** (x / 1536), 6) for x in (0, 384, 768, 1152)]
    res["win_only"] = dict(rc=rc, rew_sum_x_B=[round(x, 4) for x in sums], ent_coef=ents, betas=betas, expected_betas=exp_b)
    ok &= rc == 0 and all(abs(x - round(x)) < 1e-4 for x in sums) and ents == [0.01, 0.0075, 0.005, 0.0025] and betas == exp_b
    # (3) resume with a LONGER horizon: the curves continue from the saved values (no jump back up), VWARM_RESUME
    rc2, log2, st2, _ = run("sprintWin", dict(env, ITERS="6", HORIZON_SAMPLES="3072", RESUME=os.path.join(d, "state_4.npz"),
                                               VWARM_RESUME="1", LR_SCALE="sqrt", LR_REF_MB="4", DEVICES="2"), devices=2, fresh=False)
    s5, s6 = st2[-2], st2[-1]
    b_saved = 0.05 * 0.1 ** (1536 / 1536)                  # beta_now saved at it 4 = the value for samples 1536 (the end: 0.005)
    res["resume"] = dict(rc=rc2, beta5=s5["beta"], beta6=s6["beta"], ent5=s5["ent_coef"], ent6=s6["ent_coef"], gnorm5=s5["gnorm"],
                         gnorm6=s6["gnorm"], lr_line=[l for l in log2.splitlines() if l.startswith("[lr]")][-1:])
    ok &= rc2 == 0 and abs(s5["beta"] - 0.005) < 1e-9 and abs(s6["beta"] - 0.005) < 1e-9 and s5["ent_coef"] == 0.0
    ok &= s5["gnorm"] == 0.0 and s6["gnorm"] > 0 and "x2.828" in (res["resume"]["lr_line"] or [""])[0]   # sqrt(2 x 16 / 4)
    # (4) pt ring: exports at 384 (first), 768, 1152 (milestone 1000), 1536 (end of the first run = final), then (resumed on
    #     2 devices: 768 per it) 2304 (milestone 2000), 3072 (final): kept = first, milestones, finals + the last PT_KEEP=1
    idx = json.load(open(os.path.join(d, "pt", "index.json")))
    files = sorted(os.listdir(os.path.join(d, "pt")))
    res["pt_ring"] = dict(index=[(e["it"], e["samples"], e["keep"]) for e in idx], files=files)
    ok &= [e["it"] for e in idx if e["keep"]] == [1, 3, 4, 5, 6] and [e["it"] for e in idx if not e["keep"]] == [2]
    ok &= os.path.exists(os.path.join(d, "games.jsonl"))
    print("SPRINT", "PASS" if ok else "FAIL", json.dumps(res, default=str), flush=True)
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    oks = []
    if what in ("unit", "all"):
        oks.append(unit())
    if what in ("mockA", "all"):
        oks.append(mock_a())
    if what in ("mockB", "all"):
        oks.append(mock_b())
    if what in ("dp", "all"):
        oks.append(mock_b(devices=2, name="dpB"))
    if what in ("dpresume", "all"):
        oks.append(dp_resume())
    if what in ("sprint", "all"):
        oks.append(sprint())
    print("ALL PASS" if all(oks) else "SOME FAIL", flush=True)
