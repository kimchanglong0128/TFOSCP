#!/bin/bash
# waits for the SECOND "end N2" line (the rerun), then M1 rerun, then N1 rerun
cd /workspace/exp
until [ "$(grep -c 'end N2' cards/queue.log 2>/dev/null)" -ge 2 ]; do sleep 60; done
for c in M1 N1; do echo "=== start $c (rerun2) $(date)" >> cards/queue.log; (cd $c && python run.py > run.log 2>&1); echo "=== end $c rerun2 $(date) exit $?" >> cards/queue.log; done
