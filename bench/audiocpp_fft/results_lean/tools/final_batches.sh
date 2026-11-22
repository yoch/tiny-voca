#!/usr/bin/env bash
# The final timing batches: real flags (g++), x86-64-v3 (g++), real flags (clang++), then the checks.
W=$(cd "$(dirname "$0")/.." && pwd)
while pgrep -f "tools/batch.sh" > /dev/null; do sleep 3; done
$W/tools/batch.sh f_real 40 ref:LEAN ref:MID ref:MIDG ref:fin ref:kit fin:LEAN fin:MID fin:MIDG ref:refM A1ECD:LEAN
FLAGS=v3 $W/tools/batch.sh f_v3 30 ref:LEAN ref:MID ref:MIDG ref:fin ref:kit fin:LEAN fin:MID fin:MIDG
CXX=clang++ $W/tools/batch.sh f_clang 30 ref:LEAN ref:MID ref:MIDG ref:fin fin:LEAN fin:MID
$W/tools/checks.sh LEAN MID MIDG fin
echo ALL_DONE > $W/res/final_done.txt
