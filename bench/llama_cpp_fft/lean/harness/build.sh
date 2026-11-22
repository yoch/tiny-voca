#!/usr/bin/env bash
# build.sh OUTDIR "EXTRA_FLAGS" cand1.cpp cand2.cpp ... : every candidate in its own TU, all + fft.cpp in one -fPIC shared lib
set -e
H=$(cd "$(dirname "$0")" && pwd)
OUT=$1; shift; EXTRA=$1; shift
CXX=${CXX:-g++}
FL="-O3 -DNDEBUG -fPIC -std=c++17 -D_USE_MATH_DEFINES $EXTRA"
mkdir -p "$OUT"
objs=("$OUT/fft.o"); names=()
$CXX $FL -I"$H" -c "$H/fft.cpp" -o "$OUT/fft.o"
for f in "$@"; do
  n=$(basename "$f" .cpp); names+=("$n")
  "$H/wrap.sh" "$n" "$f" > "$OUT/tu_$n.cpp"
  $CXX $FL -I"$H" -I"$(dirname "$f")" -c "$OUT/tu_$n.cpp" -o "$OUT/tu_$n.o" &
  objs+=("$OUT/tu_$n.o")
done
wait
{ echo '#include "shim.h"'; echo '#include <map>'; for n in "${names[@]}"; do echo "extern synth_fn reg_$n;"; done
  echo 'std::map<std::string, synth_fn> & registry() { static std::map<std::string, synth_fn> m = {'
  for n in "${names[@]}"; do echo "{\"$n\", reg_$n},"; done; echo '}; return m; }'; } > "$OUT/reg.cpp"
$CXX $FL -I"$H" -c "$OUT/reg.cpp" -o "$OUT/reg.o"
$CXX -shared -o "$OUT/libcands.so" "${objs[@]}" "$OUT/reg.o"
$CXX $FL -I"$H" "$H/main.cpp" -o "$OUT/bench" -L"$OUT" -lcands -Wl,-rpath,"$OUT"
