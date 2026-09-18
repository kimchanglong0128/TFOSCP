#!/bin/bash
cd /workspace/exp
until grep -q "end D1" cards/queue.log 2>/dev/null; do sleep 60; done
for c in D2 N2 M1; do
  echo "=== start $c $(date)" >> cards/queue.log
  (cd $c && python run.py > run.log 2>&1); echo "=== end $c $(date) exit $?" >> cards/queue.log
done
