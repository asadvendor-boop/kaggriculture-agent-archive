#!/bin/bash
# Server CPU parity sweeps (1 core each via srv_jax.sh).  usage: server_chain.sh A|B   (launch detached)
R=/Users/asad.ali/kaggriculture; W="bash $R/kagjax/srv_jax.sh /opt/jaxenv/bin/python -m"
export PYTHONPATH=$R KJP_RESULTS=$R/kagjax/policy/results_server
mkdir -p $KJP_RESULTS
cd $R
if [ "$1" = A ]; then
  $W kagjax.policy.test_encode 100 1 0
  $W kagjax.policy.test_act $R/il/ckpt/fr_t_dsm_s0.pt 12 24
  $W kagjax.policy.test_selfplay $R/il/ckpt/fr_t_dsm_s0.pt 16 900001
  $W kagjax.policy.test_encode_fuzz 20480
else
  $W kagjax.policy.test_encode 100 1 1
  $W kagjax.policy.test_act $R/il/ckpt/g20_u_dsm_s0.pt 8 16
  $W kagjax.policy.test_selfplay $R/il/ckpt/g20_u_dsm_s0.pt 8 910001
  $W kagjax.policy.test_act $R/il/ckpt/rl2_it24.pt 8 16
  $W kagjax.policy.test_adapter 60 3
fi
echo CHAIN_DONE
