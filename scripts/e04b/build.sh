#!/usr/bin/env bash
set -euo pipefail
src="$(dirname "$0")/dsp.c"
out="${1:?usage: build.sh OUTPUT_DIRECTORY}"
mkdir -p "$out"
march="${E04B_MARCH:-x86-64}"
case "$march" in
  x86-64) ;;
  x86-64-v3)
    # OS-visible CPUID features required by the compiler's v3 target.
    for flag in avx avx2 bmi1 bmi2 f16c fma movbe xsave abm cx16 lahf_lm popcnt sse3 ssse3 sse4_1 sse4_2; do
      [[ "$flag" == sse3 ]] && flag=pni
      if ! awk -v name="$flag" '/^flags[[:space:]]*:/ {n++; if ($0 ~ (" " name "( |$)")) m++} END {exit (!n || m!=n)}' /proc/cpuinfo; then
        echo "Cannot build for x86-64-v3 on this host: missing $flag" >&2
        exit 1
      fi
    done
    ;;
  *) echo "Use x86-64 or x86-64-v3 for this historical benchmark" >&2; exit 1 ;;
esac
"${CC:-cc}" -O3 -march="$march" -ffp-contract=off -fPIC -shared "$src" -lm -o "$out/dsp.so"
"${PYTHON:-python}" -m ziglang cc -target wasm32-wasi -O3 -msimd128 -ffp-contract=off \
    -mexec-model=reactor -Wl,--strip-all,--export=initialize,--export=filter,--export=compact_filter,--export=oscillator,--export=mix,--export=__heap_base \
    "$src" -o "$out/dsp.wasm"
