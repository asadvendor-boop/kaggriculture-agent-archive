#!/bin/bash
# Sep 28 ~16:05 PKT, wave 3: combos of the rules that passed so far (guarded sheep sy7g1, feed guard, strawberries) + sheep dose
# 5 / 9 per yarn store; one srv_wave2.sh chain each, started only while the box has room (<= MAXW workers), 45 s apart.
K=/Users/asad.ali/kaggriculture; W=$K/il/live/srv_wave2.sh
room() { while [ $(ps -eo args | grep -c "[m]ultiprocessing.spawn") -gt ${MAXW:-200} ]; do sleep 20; done; }
go() { room; setsid -f bash $W "$@" > /root/w3_$(echo "$2_$3" | cut -d'|' -f1).out 2>&1 < /dev/null; sleep 45; }
go ilsp_v13_p3 soh "sy7g1fgstr|SHEEP_YARN=7 LAND_GUARD=1 FEED_GUARD=1 STRAW_UP=4"
go ilsp_v13_p3 soh "sy7g1fg|SHEEP_YARN=7 LAND_GUARD=1 FEED_GUARD=1"
go ilsp_v13_p3 soh "sy7g1str|SHEEP_YARN=7 LAND_GUARD=1 STRAW_UP=4"
go ilsp_v13_newplan nph "sy7g1fgstr|SHEEP_YARN=7 LAND_GUARD=1 FEED_GUARD=1 STRAW_UP=4"
go ilsp_v13_p3 soh "sy5g1|SHEEP_YARN=5 LAND_GUARD=1"
go ilsp_v13_p3 soh "sy9g1|SHEEP_YARN=9 LAND_GUARD=1"
go ilsp_v13_p3 soh "fgstr|FEED_GUARD=1 STRAW_UP=4"
go ilsp_v13_newplan nph "fgstr|FEED_GUARD=1 STRAW_UP=4"
echo "[$(date -u +%H:%M) UTC] wave3 all launched" >> /root/tonight.log
