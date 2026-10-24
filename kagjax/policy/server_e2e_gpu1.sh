#!/bin/bash
# All five end-to-end JAX runs under ONE GPU-lock acquisition (the lock queue is long; ~13 min hold).  Launch detached.
R=/Users/asad.ali/kaggriculture; O=$R/kagjax/policy/e2e_out; mkdir -p $O
export PYTHONPATH=$R
cd $R
bash $R/kagjax/srv_jax.sh --gpu bash -c "
P='/opt/jaxenv/bin/python -m kagjax.policy.e2e_jax'
\$P --learner $R/il/ckpt/rl2_it24.pt --games 64 --seed0 881000 --precision highest --out $O/rl2_hi.npz
\$P --learner $R/il/ckpt/g20_u_dsm_s0.pt --games 64 --seed0 880000 --precision highest --out $O/g20_hi.npz
\$P --learner $R/il/ckpt/g20_u_dsm_s0.pt --opp $R/il/ckpt/fr_t_s0.pt --oteam 0 --otau 0.5 --games 64 --seed0 882000 --precision highest --out $O/vs_hi.npz
\$P --learner $R/il/ckpt/rl2_it24.pt --games 64 --seed0 881000 --precision default --shadow --out $O/rl2_tf32.npz
\$P --learner $R/il/ckpt/g20_u_dsm_s0.pt --games 64 --seed0 880000 --precision default --shadow --out $O/g20_tf32.npz
"
echo E2E_GPU_DONE
