#!/usr/bin/env bash
# Final harness timings: reference / kit opt / final candidate, interleaved (cmp.py, 15 rounds x 15 runs, core 2).
set -u
W=$(cd "$(dirname "$0")/.." && pwd); cd $W
REF=audio.cpp/src/community_models/tiny_voca/source_filter.cpp
b() { # name compiler opt hdr src [defines]
  CXX=$2 OPT="$3" HDR=$4 tools/build.sh $1 $5 ${6:-}
}
for cfg in "gn|g++|-O3 -march=native" "g0|g++|-O3" "cn|clang++|-O3 -march=native" "c0|clang++|-O3"; do
  IFS='|' read nm cxx o <<< "$cfg"
  for mt in nomt mt; do
    d=""; [ $mt = nomt ] && d=-DENGINE_SPEECHFFT_NO_MULTITHREADING
    [ $nm != gn ] && [ $mt = mt ] && continue     # the threaded FFT is measured with the main config only
    b ref_${nm}_$mt $cxx "$o" $W/../stub $REF $d
    b kit_${nm}_$mt $cxx "$o" $W/../opt/include ../opt/src/source_filter.cpp $d
    b fin_${nm}_$mt $cxx "$o" $W/fin/include cand.cpp $d
    echo "=== $cxx $o $d"
    python3 tools/cmp.py 15 15 ref_${nm}_$mt kit_${nm}_$mt fin_${nm}_$mt
  done
done
echo "=== variant target_clones, g++ -O3 (no -march) -DENGINE_SPEECHFFT_NO_MULTITHREADING"
b clones_g0 g++ "-O3" $W/fin/include ../variants/target_clones/source_filter.cpp -DENGINE_SPEECHFFT_NO_MULTITHREADING
python3 tools/cmp.py 15 15 fin_g0_nomt clones_g0
echo FINAL_DONE
