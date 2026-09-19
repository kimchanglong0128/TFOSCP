#!/bin/bash
cd /workspace/exp
until grep -q "end MB2" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start FO1 $(date)" >> cards/queue.log; (cd FO1 && python run.py > run.log 2>&1); echo "=== end FO1 $(date) exit $?" >> cards/queue.log
