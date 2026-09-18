#!/bin/bash
cd /workspace/exp
until grep -q "end M1" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start N1 $(date)" >> cards/queue.log; (cd N1 && python run.py > run.log 2>&1); echo "=== end N1 $(date) exit $?" >> cards/queue.log
