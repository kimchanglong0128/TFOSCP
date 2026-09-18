#!/bin/bash
cd /workspace/exp
until grep -q "end N2 rerun3" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start R1 $(date)" >> cards/queue.log; (cd R1 && python run.py > run.log 2>&1); echo "=== end R1 $(date) exit $?" >> cards/queue.log
