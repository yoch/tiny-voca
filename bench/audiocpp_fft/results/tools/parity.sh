#!/usr/bin/env bash
# parity.sh NAME : writes ../out/NAME and compares it to the reference output
W=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $W/out/$1
$W/bin/$1 $W/dumps 1 $W/out/$1 > /dev/null && $W/venv/bin/python $W/../compare.py $W/out_ref $W/out/$1 | awk '{print} {s=$5; if (min=="" || s<min) min=s; if ($10>mx) mx=$10; n+=$12} END {printf "PARITY min SNR %s dB, max diff %s LSB, %d samples differ\n", min, mx, n}'
