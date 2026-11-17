#!/usr/bin/env bash
set -euo pipefail
src="$(dirname "$0")/dsp.c"
out="${1:?usage: build.sh OUTPUT_DIRECTORY}"
mkdir -p "$out"
"${CC:-cc}" -O3 -march="${E04B_MARCH:-x86-64-v3}" -ffp-contract=off -fPIC -shared "$src" -lm -o "$out/dsp.so"
"${PYTHON:-python}" -m ziglang cc -target wasm32-wasi -O3 -msimd128 -ffp-contract=off \
    -mexec-model=reactor -Wl,--strip-all,--export=initialize,--export=filter,--export=compact_filter,--export=oscillator,--export=mix,--export=__heap_base \
    "$src" -o "$out/dsp.wasm"
