#!/bin/bash
# Mac rule (operator, Sep 28 night): at most 4 heavy Python processes in total -> wait until fewer than 4 run.
while true; do
  n=$(ps -axo command | grep -c "[P]ython.app/Contents/MacOS/Python")
  [ "$n" -lt 4 ] && { echo "slots: $n heavy Python running -> go"; exit 0; }
  echo "slots: $n heavy Python running -> waiting 30 s"; sleep 30
done
