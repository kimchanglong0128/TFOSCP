#!/bin/bash
cd /workspace/exp
until grep -q "end BG1" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start MB2 $(date)" >> cards/queue.log; (cd MB2 && python run.py > run.log 2>&1); echo "=== end MB2 $(date) exit $?" >> cards/queue.log
