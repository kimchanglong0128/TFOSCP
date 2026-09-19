#!/bin/bash
# Part 2 of the Drive upload. The shared rclone API project hit its per-minute quota (403 Quota exceeded), and a streamed rcat cannot be retried.
# So: write each card's tar to the volume first, then `rclone copy` it (retriable) in a loop until it succeeds, then delete the local tar.
cd /workspace; LOG=/workspace/exp/cards/drive_upload.log; TMP=/workspace/tmp_tars; mkdir -p $TMP
R="--drive-chunk-size 256M --tpslimit 2 --retries 5 --retries-sleep 60s --low-level-retries 20"
DONE=" AD9 AE2 AE3 B1 B1b B2 B2b D1 D1b "
up() {  # up <local file or dir> <remote dir> [extra rclone args]; retries up to 30 times, 2 min apart
  local src=$1 dst=$2; shift 2
  for i in $(seq 1 30); do rclone copy "$src" "$dst" $R "$@" >> $LOG 2>&1 && return 0; echo "retry $i for $src $(date)" >> $LOG; sleep 120; done; return 1
}
for d in exp/*/gen; do c=$(basename $(dirname $d)); case "$DONE" in *" $c "*) continue;; esac
  echo "=== gen $c $(date)" >> $LOG; tar -chf $TMP/${c}_gen.tar -C exp/$c gen && up $TMP/${c}_gen.tar gdrive:OSCM/gen_tars; rc=$?; [ $rc -eq 0 ] && rm -f $TMP/${c}_gen.tar; echo "exit $rc $c $(date)" >> $LOG; done
echo "=== browsable text $(date)" >> $LOG; up exp gdrive:OSCM/exp --include "LOG.md" --include "cards/STATUS.md" --include "*/summary*.json" --include "R1/budget.json" --transfers 2; echo "exit $? text $(date)" >> $LOG
echo "=== outputs (resume) $(date)" >> $LOG; up outputs gdrive:OSCM/outputs --transfers 2; echo "exit $? outputs $(date)" >> $LOG
rmdir $TMP 2>/dev/null; echo "ALL DONE $(date)" >> $LOG
