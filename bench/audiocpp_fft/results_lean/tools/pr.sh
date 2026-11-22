#!/usr/bin/env bash
# pr.sh A B ITERS [MODE]: paired in-process timing of abl/A (A) against abl/B (B) with tools/pair.cpp.
# FLAGS=real (default): how audio.cpp builds the file, -O3 -fopenmp, no -march, threaded framework FFT.
# FLAGS=v3: the same plus -march=x86-64-v3 (AVX2 + FMA). CORE selects the pinned core (default 2).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); B=$W/..; A=$W/audio.cpp
a=$1; b=$2; iters=$3; mode=${4:-}
case ${FLAGS:-real} in
  real) F="-O3 -fopenmp" ;;
  v3) F="-O3 -fopenmp -march=x86-64-v3" ;;
  *) F="$FLAGS" ;;
esac
F="$F -std=c++17 -DNDEBUG"
T=$W/pairbuild/${a}_${b}_${FLAGS:-real}_${CXX:-g++}; mkdir -p $T $W/bin
for x in a b; do
  v=$a; [ $x = b ] && v=$b
  { grep -v "pragma once" $B/stub/engine/community_models/tiny_voca/assets.h; grep -v "pragma once\|tiny_voca/assets.h" $W/abl/$v/include/engine/community_models/tiny_voca/source_filter.h; } > $T/hdr_$x.h
done
${CXX:-g++} $F -Dtiny_voca=tv_a -I$W/abl/$a/include -I$B/stub -I$A/include -I$A/src/framework/audio -c $W/abl/$a/src/source_filter.cpp -o $T/a.o &
${CXX:-g++} $F -Dtiny_voca=tv_b -I$W/abl/$b/include -I$B/stub -I$A/include -I$A/src/framework/audio -c $W/abl/$b/src/source_filter.cpp -o $T/b.o &
[ -f $T/fft.o ] || ${CXX:-g++} $F -I$A/include -I$A/src/framework/audio -c $A/src/framework/audio/fft.cpp -o $T/fft.o &
${CXX:-g++} $F -I$T -I$B/stub -c $W/tools/pair.cpp -o $T/pair.o &
wait
${CXX:-g++} ${F/-fopenmp/} $T/a.o $T/b.o $T/fft.o $T/pair.o -lpthread -o $T/pair  # linked without -fopenmp: no OpenMP runtime is used, and clang has none installed here
if [ -n "${NOPIN:-}" ]; then $T/pair $W/dumps $iters $mode; else taskset -c ${CORE:-2} $T/pair $W/dumps $iters $mode; fi
