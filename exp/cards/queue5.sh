#!/bin/bash
cd /workspace/exp
until grep -q "end S1b" cards/queue.log 2>/dev/null; do sleep 60; done
echo "=== start D1b $(date)" >> cards/queue.log; (cd D1b && python run.py > run.log 2>&1); echo "=== end D1b $(date) exit $?" >> cards/queue.log
