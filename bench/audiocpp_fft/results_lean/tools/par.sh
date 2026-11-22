#!/usr/bin/env bash
# par.sh NAME [REFOUT]: builds abl/NAME with the real flags (-O3 -fopenmp, threaded FFT) into bin/r_NAME, writes
# out/r_NAME and compares it with REFOUT (default out/r_ref, the reference built the same way).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); n=$1; ref=${2:-$W/out/r_ref}
CXX=${CXX:-g++} OPT="${OPT:--O3 -fopenmp}" HDR=$W/abl/$n/include $W/tools/build.sh r_$n $W/abl/$n/src/source_filter.cpp
mkdir -p $W/out/r_$n
$W/bin/r_$n $W/dumps 1 $W/out/r_$n > /dev/null
$W/venv/bin/python $W/../compare.py $ref $W/out/r_$n | awk '{s=$5; if (min=="" || s<min) min=s; if ($10>mx) mx=$10; n+=$12} END {printf "PARITY min SNR %s dB, max diff %s LSB, %d samples differ\n", min, mx, n}'
