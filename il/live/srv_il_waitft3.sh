#!/bin/bash
# Sep 27 ~15:02 UTC: when the pass stage of <src> has finished (its log shows the DSM fine-tune start, so
# ckpt/ilsp_<src>_p.pt is final), run a 3-epoch DSM fine-tune from it on the first free GPU (srv_il_ft3.sh).
#   srv_il_waitft3.sh <src> <name>
cd /Users/asad.ali/kaggriculture/il
until grep -q "DSM fine-tune" logs/ilsp_$1.log; do sleep 15; done
free_gpu() { local g p b; for g in 0 1 2 3 4 5 6 7; do b=0; for p in $(pgrep -f "python train6.py"); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" && b=1; done; [ $b = 0 ] && { echo $g; return; }; done; }
while [ -z "$(free_gpu)" ]; do sleep 10; done
bash live/srv_il_ft3.sh $(free_gpu) $2 ckpt/ilsp_$1_p.pt 0
