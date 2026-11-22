#!/usr/bin/env bash
# batch.sh TAG ITERS A:B ... : runs pr.sh for each pair, appends the TOTAL lines to res/TAG.txt
W=$(cd "$(dirname "$0")/.." && pwd); tag=$1; it=$2; shift 2
for p in "$@"; do
  a=${p%%:*}; b=${p##*:}
  r=$($W/tools/pr.sh $a $b $it 2>&1 | tail -1)
  echo "$a -> $b [${FLAGS:-real}]: $r" | tee -a $W/res/$tag.txt
done
echo BATCH_DONE >> $W/res/$tag.txt
