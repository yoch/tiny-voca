# Reproduce E04-C

The pilot is opt-in and only runs with a matched asset bundle. Use the pinned
checkpoint, frontend data, Python and browser package versions from the
[E04-B setup](../e04b/REPRODUCE.md). The dense reference and pilot weights
come from the same checkpoint. Run the browser suites sequentially.

## Build the bundle

```sh
export PYTHON=/tmp/e04-venv/bin/python DIZIONE_DATA=/tmp/e04-phonemizer
export ZIG_GLOBAL_CACHE_DIR=/tmp/e04c-zig
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv pip install --python "$PYTHON" ziglang==0.15.1
mkdir -p /tmp/e04c/assets
bash web/build_dsp.sh /tmp/e04c/dsp.wasm
curl -fL https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/config.json -o /tmp/e04c/config.json
"$PYTHON" scripts/e04c/export_bundle.py \
  --checkpoint /tmp/e04-checkpoint.pt \
  --checkpoint-sha256 ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70 \
  --config /tmp/e04c/config.json --wasm /tmp/e04c/dsp.wasm \
  --output /tmp/e04c/assets --test-reference
(cd /tmp/e04-dizione/src && zip -q -r /tmp/e04c/assets/dizione.zip dizione -x '*/__pycache__/*')
cp docs/e04/e04c/cases.json /tmp/e04c/cases.json
```

The public `web/fetch_assets.sh` remains the normal dense bundle. To stage a
pilot, run it first for the normal frontend data, then build the verified
bundle into `web/public/assets` without `--test-reference`. Host the directory
on a secure origin. Use the normal URL for dense, or `?vocoder=compact` for
the pilot. Four threads additionally require `&threads=4`, COOP
`same-origin`, COEP `require-corp` and a compatible browser. The pilot does
not add these headers to the existing deployment.

## Tests and measurements

```sh
bash tests/dsp/run.sh
TINY_VOCA_TEST_WASM=/tmp/e04c/dsp.wasm node --test tests/dsp/wasm.cjs
"$PYTHON" -m pytest tests/test_export.py tests/test_fft_candidates.py -q
# Node packages and Playwright data as in E04-B.
export NODE_PATH=/tmp/e04-web/node_modules PLAYWRIGHT_BROWSERS_PATH=/tmp/e04-browsers
/tmp/e04-web/node_modules/.bin/playwright install chromium firefox webkit
taskset -c 0-3 node scripts/e04c/browser.cjs /tmp/e04c
"$PYTHON" scripts/e04c/listening.py /tmp/e04c docs/e04/e04b/audio /tmp/e04c/listening
```

The browser harness requires the system libraries for each engine, a writable
Firefox profile/config, and `DIZIONE_DATA`. The optional `E04C_BROWSERS` list
selects engines sequentially. Use `E04C_REPORT` to name independent result
files. `E04C_WEBKIT_EXECUTABLE` can point to a runner-specific WebKit
launcher. Keep such infrastructure overrides out of the site. The blind WAV
archive excludes `key.json`. Keep the separate key closed until scores
have been recorded.

## Native and French parity

For a Linux x86-64 baseline use a generic build. A separately compiled v3
artifact must only run on a host supporting that ISA.
The native checker does not select a library automatically.

```sh
cc -march=x86-64 -O3 -ffp-contract=off -fPIC -shared \
  src/tiny_voca/vocoder_dsp.c -lm -o /tmp/e04c/native-x86-64.so
"$PYTHON" scripts/e04c/native.py --checkpoint /tmp/e04-checkpoint.pt \
  --checkpoint-sha256 ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70 \
  --library /tmp/e04c/native-x86-64.so --output /tmp/e04c/en

git clone https://github.com/DanielSuncost/tiny-voca-fr /tmp/e04c-fr
git -C /tmp/e04c-fr checkout 4cc26ab23d934e3d33796eb06bf11f23f4c9ab34
curl -fL https://huggingface.co/DanielSuncost/tiny-voca-fr/resolve/73f7700f514e70eef3b459aedcc49829d41945d4/tiny-voca-fr.pt -o /tmp/e04c/fr.pt
# Ensure eSpeak NG is on PATH.
PYTHONPATH=/tmp/e04c-fr/src TINY_VOCA_LANGUAGE=fr "$PYTHON" scripts/e04c/native.py \
  --checkpoint /tmp/e04c/fr.pt \
  --checkpoint-sha256 61c00f50dffeda1cc8773243b4722a1d64b57a016db90ecd96477aae6225b3a9 \
  --library /tmp/e04c/native-x86-64.so --output /tmp/e04c/fr --language fr
```

A QEMU Nehalem run validates the baseline C executable without AVX2.
A statically cross-compiled C executable validates AArch64 under QEMU.
These are functional C tests rather than timing or whole-device evidence.
