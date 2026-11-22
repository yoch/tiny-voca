#!/usr/bin/env bash
# build.sh NAME SOURCE.cpp [extra flags...]   -> ../bin/NAME   (CXX env selects the compiler)
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); B=$W/..; A=$W/audio.cpp
name=$1; src=$2; shift 2
mkdir -p $W/bin
${CXX:-g++} ${OPT:--O3 -march=native} -std=c++17 -DNDEBUG "$@" -I${HDR:-$B/opt/include} -I$B/stub -I$A/include -I$A/src/framework/audio \
  $src $B/bench_ac.cpp $A/src/framework/audio/fft.cpp -lpthread -o $W/bin/$name
