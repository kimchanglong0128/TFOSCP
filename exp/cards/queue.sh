#!/bin/bash
# sequential queue: waits for B1, then runs B2 -> T1 -> D1 (each logs to its own run.log)
cd /workspace/exp
until grep -q "^DONE\|Traceback\|AssertionError" B1/run.log; do sleep 30; done
for c in B2 T1 D1; do
  echo "=== start $c $(date)" >> cards/queue.log
  (cd $c && python run.py > run.log 2>&1); echo "=== end $c $(date) exit $?" >> cards/queue.log
done
