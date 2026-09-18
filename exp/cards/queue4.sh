#!/bin/bash
cd /workspace/exp
until grep -q "end N1" cards/queue.log 2>/dev/null; do sleep 60; done
for c in B1b B2b; do echo "=== start $c $(date)" >> cards/queue.log; (cd $c && python run.py > run.log 2>&1); echo "=== end $c $(date) exit $?" >> cards/queue.log; done
echo "=== start S1b $(date)" >> cards/queue.log; (cd S1 && python run_b.py > run_b.log 2>&1); echo "=== end S1b $(date) exit $?" >> cards/queue.log
