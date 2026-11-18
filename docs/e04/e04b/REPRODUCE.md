# Reproduce E04-B

Start from the draft branch and use the public checkpoint, pinned Python packages,
dizione revision and prebuilt dictionaries in the [E04 AVX2 setup](../avx2/REPORT.md#reproduce).
GPU is not needed. The checkpoint SHA256 is checked again by the preparation script.
Keep outputs outside the checkout.

Prepare the original E04 data if reproducing the CPU operator trace or the ISA
microbenchmark. The old benchmark command supports `--runs 1` for that setup.
The profile uses the old 194-frame phone case and generates inspector trace
files plus a compact node table. The end-to-end runs reuse only the checkpoint.

```sh
export PYTHON=/tmp/e04-venv/bin/python
export DIZIONE_DATA=/tmp/e04-phonemizer TINY_VOCA_FRONTEND=own
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ZIG_GLOBAL_CACHE_DIR=/tmp/e04b-zig
uv pip install --python "$PYTHON" ziglang==0.15.1
"$PYTHON" scripts/e04b/prepare.py /tmp/e04-checkpoint.pt /tmp/e04b
E04B_MARCH=x86-64-v3 bash scripts/e04b/build.sh /tmp/e04b

npm install --prefix /tmp/e04-web onnxruntime-web@1.20.1 playwright@1.65.1 pyodide@0.26.4
export NODE_PATH=/tmp/e04-web/node_modules PLAYWRIGHT_BROWSERS_PATH=/tmp/e04-browsers
/tmp/e04-web/node_modules/.bin/playwright install chromium
wheel=numpy-1.26.4-cp312-cp312-pyodide_2024_0_wasm32.whl
curl -fL "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/$wheel" \
  -o "/tmp/e04-web/node_modules/pyodide/$wheel"
echo "4a2f5303a88a0747c5e6c80701e0ff4ff5667e487c0c9f9c73875e4d88f4cf9a  /tmp/e04-web/node_modules/pyodide/$wheel" | sha256sum -c -

# Select four permitted logical CPUs on your runner. Execute suites sequentially.
taskset -c 0-3 "$PYTHON" scripts/e04b/cpu.py /tmp/e04b
taskset -c 0-3 "$PYTHON" scripts/e04b/cpu.py /tmp/e04b --reverse
taskset -c 0-3 node scripts/e04b/web.cjs /tmp/e04b
taskset -c 0-3 node scripts/e04b/web.cjs /tmp/e04b --reverse
taskset -c 0-3 "$PYTHON" scripts/e04b/cpu.py /tmp/e04b \
  --thread-counts 4 --variants dense fused fused2 --result-prefix parallel
taskset -c 0-3 "$PYTHON" scripts/e04b/cpu.py /tmp/e04b --reverse \
  --thread-counts 4 --variants dense fused fused2 --result-prefix parallel
"$PYTHON" scripts/e04b/summarize.py /tmp/e04b > /tmp/e04b/summary.json
"$PYTHON" scripts/e04b/verify.py /tmp/e04b
"$PYTHON" -m pytest tests/test_export.py tests/test_fft_candidates.py -q
taskset -c 0 "$PYTHON" scripts/e04b/roofline.py > /tmp/e04b/roofline.json
taskset -c 0 "$PYTHON" scripts/e04b/isa.py /tmp/e04b /tmp/e04-avx2/full-dense.npz > /tmp/e04b/isa.json
taskset -c 0-3 "$PYTHON" scripts/e04b/profile.py /tmp/e04-avx2 /tmp/e04b/profile
```

The optional early boundary is a diagnostic variant. On CPU use `--variants
dense early`. On Web use `E04B_VARIANTS=dense,early`. Point the diagnostic
at a separate output directory with the prepared artifacts to preserve the
acceptance run files. The early variant fails parity due to phase drift.

`cpu-audio/` and `web-audio/` contain 24 kHz float WAVs for all tested
durations, voices and thread counts. Each filename pairs a dense control and
a candidate using the same thread count, case, seed and runtime. For a
listening test, apply the same peak gain to both WAVs, randomize A/B/X labels,
and cover onset, pitched transitions, unvoiced consonants and the end of long
utterances. Numerical parity is checked on the raw audio before limiting.
