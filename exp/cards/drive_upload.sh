#!/bin/bash
# Upload everything to gdrive:OSCM. rclone's shared API key is rate-limited per request, so bulk data goes as a few large tar streams;
# only the files worth browsing online (outputs/, LOG, STATUS, summaries) are copied one by one, last, with a request-rate cap.
cd /workspace; LOG=/workspace/exp/cards/drive_upload.log; : > $LOG; R="--drive-chunk-size 256M --tpslimit 4 --retries 10 --low-level-retries 20"
echo "=== exp_nogen.tar $(date)" >> $LOG; tar -cf - --exclude='gen' --exclude='__pycache__' exp | rclone rcat gdrive:OSCM/exp_nogen.tar $R >> $LOG 2>&1; echo "exit $? exp_nogen $(date)" >> $LOG
for d in exp/*/gen; do c=$(basename $(dirname $d)); echo "=== gen $c $(date)" >> $LOG; tar -chf - -C exp/$c gen | rclone rcat gdrive:OSCM/gen_tars/${c}_gen.tar $R >> $LOG 2>&1; echo "exit $? $c $(date)" >> $LOG; done
echo "=== browsable text $(date)" >> $LOG; rclone copy exp gdrive:OSCM/exp --include "LOG.md" --include "cards/STATUS.md" --include "*/summary*.json" --include "R1/budget.json" --transfers 2 $R >> $LOG 2>&1; echo "exit $? text" >> $LOG
echo "=== outputs (resume) $(date)" >> $LOG; rclone copy outputs gdrive:OSCM/outputs --transfers 2 $R >> $LOG 2>&1; echo "exit $? outputs" >> $LOG
echo "ALL DONE $(date)" >> $LOG
