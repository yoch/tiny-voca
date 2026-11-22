#!/usr/bin/env bash
# checks.sh NAME...: pulse train accuracy and short inputs (1..590 frames) of each candidate against the reference,
# real flags (g++ -O3 -fopenmp), written to res/checks_NAME.txt
W=$(cd "$(dirname "$0")/.." && pwd)
while pgrep -f "tools/batch.sh" > /dev/null; do sleep 3; done
for n in "$@"; do
  { echo "== $n pulse_train vs reference"; $W/tools/pr.sh ref $n 1 pulse; echo "== $n short inputs vs reference"; $W/tools/pr.sh ref $n 1 edge; } > $W/res/checks_$n.txt 2>&1
done
echo CHECKS_DONE >> $W/res/checks_done.txt
