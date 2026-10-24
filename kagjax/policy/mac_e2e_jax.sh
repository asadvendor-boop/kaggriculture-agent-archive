#!/bin/bash
# JAX side of the end-to-end identity test on the Mac CPU (fp32), sequential, 1 core; then compare with the references.
R=/Users/asad.ali/kaggriculture; P=$R/kagjax/policy; O=$P/e2e_out; F=$P/e2e_ref; mkdir -p $O
$P/run1.sh -m kagjax.policy.numpy_xplat run $R/il/ckpt/rl2_it24.pt $P/results_server/xplat_mac.npz
$P/run1.sh -m kagjax.policy.numpy_xplat cmp $P/results_server/xplat_mac.npz $P/results_server/xplat_linux.npz
$P/run1.sh -m kagjax.policy.e2e_jax --learner $R/il/ckpt/rl2_it24.pt --games 64 --seed0 881000 --precision highest --out $O/cpu_rl2_hi.npz
$P/run1.sh -m kagjax.policy.e2e_compare --jax $O/cpu_rl2_hi.npz --ref $F/rl2_ref.json.gz --out $O/cmp_cpu_rl2_hi.json
$P/run1.sh -m kagjax.policy.e2e_jax --learner $R/il/ckpt/g20_u_dsm_s0.pt --games 64 --seed0 880000 --precision highest --out $O/cpu_g20_hi.npz
$P/run1.sh -m kagjax.policy.e2e_compare --jax $O/cpu_g20_hi.npz --ref $F/g20_ref.json.gz --out $O/cmp_cpu_g20_hi.json
$P/run1.sh -m kagjax.policy.e2e_jax --learner $R/il/ckpt/g20_u_dsm_s0.pt --opp $R/il/ckpt/fr_t_s0.pt --oteam 0 --otau 0.5 --games 64 --seed0 882000 --precision highest --out $O/cpu_vs_hi.npz
$P/run1.sh -m kagjax.policy.e2e_compare --jax $O/cpu_vs_hi.npz --ref $F/vs_ref.json.gz --out $O/cmp_cpu_vs_hi.json
echo MAC_JAX_DONE
