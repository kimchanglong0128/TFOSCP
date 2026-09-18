#!/bin/bash
cd /workspace/exp
until grep -q "end R1" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start N2 (rerun4) $(date)" >> cards/queue.log; (cd N2 && python run.py > run.log 2>&1); echo "=== end N2 rerun4 $(date) exit $?" >> cards/queue.log
