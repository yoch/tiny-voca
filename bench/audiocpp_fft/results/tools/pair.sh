#!/usr/bin/env bash
# pair.sh A.cpp HDR_A B.cpp HDR_B ITERS [flags...]: builds ../bin/pair and runs it pinned to one core.
# HDR_x is the directory holding engine/community_models/tiny_voca/source_filter.h for that source.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); B=$W/..; A=$W/audio.cpp; T=$W/pairbuild; mkdir -p $T
a=$1; ha=$2; b=$3; hb=$4; iters=$5; shift 5
for x in a b; do
  h=$ha; [ $x = b ] && h=$hb
  { grep -v "pragma once" $B/stub/engine/community_models/tiny_voca/assets.h; grep -v "pragma once\|tiny_voca/assets.h" $h/engine/community_models/tiny_voca/source_filter.h; } > $T/hdr_$x.h
done
F="${OPT:--O3 -march=native} -std=c++17 -DNDEBUG $*"; FB="$F ${BFLAGS:-}"
${CXX:-g++} $F -Dtiny_voca=tv_a -I$ha -I$B/stub -I$A/include -I$A/src/framework/audio -c $a -o $T/a.o &
${CXX:-g++} $FB -Dtiny_voca=tv_b -I$hb -I$B/stub -I$A/include -I$A/src/framework/audio -c $b -o $T/b.o &
${CXX:-g++} $F -I$A/include -I$A/src/framework/audio -c $A/src/framework/audio/fft.cpp -o $T/fft.o &
${CXX:-g++} $F -I$T -I$B/stub -c $W/tools/pair.cpp -o $T/pair.o &
wait
${CXX:-g++} $T/a.o $T/b.o $T/fft.o $T/pair.o -lpthread -o $W/bin/pair
taskset -c 2 $W/bin/pair $W/dumps $iters ${MODE:-}
