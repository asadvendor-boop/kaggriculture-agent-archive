#!/bin/bash
# Reference side of the end-to-end test on the Mac (1 core, sequential): C++ engine + the real exported packages.
R=/Users/asad.ali/kaggriculture; P=$R/kagjax/policy; O=$P/e2e_ref
$P/run1.sh -m kagjax.policy.e2e_ref --learner_pkg $P/packages/RL24_SE/main.py --games 64 --seed0 881000 --out $O/rl2_ref.json.gz
$P/run1.sh -m kagjax.policy.e2e_ref --learner_pkg $R/package/NEURAL_20M_SE_B64/main.py --games 64 --seed0 880000 --out $O/g20_ref.json.gz
$P/run1.sh -m kagjax.policy.e2e_ref --learner_pkg $R/package/NEURAL_20M_SE_B64/main.py --opp_pkg $P/packages/FRTS0_T0_SE/main.py --games 64 --seed0 882000 --out $O/vs_ref.json.gz
echo MAC_REF_DONE
