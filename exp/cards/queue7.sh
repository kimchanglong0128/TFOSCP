#!/bin/bash
cd /workspace/exp
until grep -q "end N2 " cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start M1 (rerun) $(date)" >> cards/queue.log; (cd M1 && python run.py > run.log 2>&1); echo "=== end M1 rerun $(date) exit $?" >> cards/queue.log
