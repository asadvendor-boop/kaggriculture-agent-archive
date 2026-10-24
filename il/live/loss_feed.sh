#!/bin/bash
# Every 15 minutes: the new live losses of both active submissions analysed on the Mac (loss_watch.py): exact C++
# replay, land/sheep/cows vs the opponent, where they out-earned us. Sep 27: the A100 server is stopped, so no push of
# rows/tapes to a server any more (the losses stay in data/tapes_n20mw*/ for the next machine's league opponents).
# Active pair since Sep 27: NEURAL_20MW_SE = 56589891 (data/tapes_n20mw), NEURAL_20MW5_SE = 56595847 (data/tapes_n20mw5).
K=/Users/asad.ali/kaggriculture
while true; do
  nice -n 10 $K/.venv/bin/python $K/il/live/loss_watch.py 56589891:$K/data/tapes_n20mw 56595847:$K/data/tapes_n20mw5 2>&1 | grep -v Warning
  sleep 900
done
