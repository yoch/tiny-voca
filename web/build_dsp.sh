#!/usr/bin/env bash
set -euo pipefail
src="$(dirname "$0")/../src/tiny_voca/vocoder_dsp.c"
out="${1:?usage: build_dsp.sh PATH_TO_WASM}"
mkdir -p "$(dirname "$out")"
"${PYTHON:-python}" -m ziglang cc -target wasm32-wasi -O3 -msimd128 -ffp-contract=off \
  -mexec-model=reactor -Wl,-z,stack-size=65536 \
  -Wl,--strip-all,--export=tv_context_size,--export=tv_initialize,--export=tv_render,--export=__heap_base \
  "$src" -o "$out"
