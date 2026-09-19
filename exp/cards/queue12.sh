#!/bin/bash
cd /workspace/exp
until grep -aq "^DONE\|Traceback\|AssertionError" MB1/run.log; do sleep 60; done
echo "=== start BG1 $(date)" >> cards/queue.log; (cd BG1 && python run.py > run.log 2>&1); echo "=== end BG1 $(date) exit $?" >> cards/queue.log
