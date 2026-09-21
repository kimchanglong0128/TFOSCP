#!/bin/bash
cd /workspace/exp/G1; python run_budget.py > run_budget.log 2>&1
for t in b_lo30_mid30 b_lo15_mid30 b_lo0_mid30 b_lo0_mid45 b_lo30_mid0; do python eval.py $t > eval_$t.log 2>&1; done; echo ALLDONE >> run_budget.log
