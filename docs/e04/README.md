# E04 — FFT vocoder experiments and handoff

[E04-C](e04c/REPORT.md) qualifies the compact backend in the production Web
worker behind an opt-in flag, with dense fallback, checked native/WASM DSP and
platform-specific results. It is a pilot, pending blind listening.

[E04-B](e04b/REPORT.md) measures a shared native and WASM FFT core with two
ONNX boundaries on real text. It includes operator profiles, repeated latency
measurements, same-runtime fidelity and comparative WAVs. The status below
describes the prior investigations.

Status: **no production change accepted**. The AVX2 rerun is complete: DFT and
radix-2 still regress on CPU and Web. A staged native CPU FFT prototype has also
been measured, but has no Web implementation or listening validation.
See [AVX2 results and reproduction](avx2/REPORT.md), including both run orders,
raw timings, fidelity checks and the decision to retain dense production export.
`src/tiny_voca/export.py` is unchanged. Both candidates are isolated in
`scripts/fft_candidates.py`; neither is selected by the production exporter.
No retraining, checkpoint mutation, frontend change or merge.

## Findings on the original runner

Intel Core i7-3770, 4 cores / 8 logical CPUs, AVX, **no AVX2**. Linux x86_64,
Python 3.14.7, PyTorch 2.15.0+cpu, ONNX 1.23.2, ONNX Runtime CPU 1.32.0.
PyTorch reports NNPACK unsupported; this warning alone does not explain ORT
performance. There is no evidence that AVX2 will reverse the result.

Two implementations replace the dense analysis/synthesis convolutions:

* `dft`: standard ONNX DFT, forward real FFT + full Hermitian inverse (opsets
  17 and 20 have different axis signatures). CPU/WASM use no custom operators.
* `radix2`: fixed-size real-valued butterflies, bit reversal and float32
  twiddles; only standard ONNX operators. Reject non-power-of-two windows.

Both use gather for framing and four-way overlap-add (reshape/slice/pad/add)
for the configured 4x overlap. Preserve left/right padding and crop exactly.
The small cepstrum-to-response projections remain dense: these are truncated
transforms and were not the target of this experiment.

Preliminary CPU measurements (median milliseconds, synthetic model):

| Workload | Threads | Dense | DFT | Radix-2 |
| --- | ---: | ---: | ---: | ---: |
| Two DSP filters, 48,128 samples | 1 | 29.62 | 56.66 | 275.65 |
| Two DSP filters, 48,128 samples | 4 | 15.37 | 48.17 | see raw data |

These are exploratory runs, with some overlapping validation/benchmark activity;
**do not use them to claim precise speedups or an AVX2 comparison**. The direction
is a regression, so no optimized production path was retained. Raw JSON files
are included for inspection; their `peak_rss_mib` values are invalid because the
original measurement used an inherited process high-water mark. Use the corrected
benchmark (Linux `/proc/self/status` VmHWM, fresh subprocess per case) for memory.
`cpu-trained-preliminary.json` uses the published trained checkpoint, unchanged.

Numerical tests compare identical deterministic noise, responses and durations.
They require filter error <= 3e-6 absolute with 2e-5 relative tolerance, whole
ONNX output normalized peak error < 2e-5 and SNR > 100 dB against dense ONNX.
Test identity, zero and edge impulses, both FFT sizes, variable lengths, voices,
speed and pitch controls. Native Torch identity filtering is checked separately:
its center padding has two fewer right-hand frames, so arbitrary responses at
the tail must not be blindly compared across those two framing conventions.
Passing numerical tests is evidence of numerical fidelity, not a listening study.

## Reproduce

Use a fresh checkout of branch `codex/e04-fft-vocoder`. No artifacts in `/tmp`
are required: the scripts generate graphs, inputs and reference outputs.
Run benchmarks sequentially on an otherwise idle runner.

```sh
uv venv --python 3.12 /tmp/e04-venv
UV_CACHE_DIR=/tmp/e04-uv uv pip install --python /tmp/e04-venv/bin/python \
  -e . -r docs/e04/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 /tmp/e04-venv/bin/python -m pytest tests/test_export.py tests/test_fft_candidates.py -q
curl -fL https://huggingface.co/n4ze3m/tiny-voca/resolve/main/tiny-voca.pt -o /tmp/e04-checkpoint.pt
sha256sum /tmp/e04-checkpoint.pt
OMP_NUM_THREADS=1 /tmp/e04-venv/bin/python scripts/benchmark_fft.py \
  --checkpoint /tmp/e04-checkpoint.pt --output /tmp/e04 --runs 30
```

Checkpoint SHA256 used here:
`ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70`.
The frontend fingerprint check is bypassed **only for direct token benchmarking**;
weights and token inventory are loaded strictly. Do not use this to bypass frontend
matching in actual text synthesis. The benchmark uses 24 deterministic phone IDs,
voice M2; expand coverage to representative sentences and both voices for acceptance.
The synthetic run (omit `--checkpoint`) is a debugging fixture, not an acoustic eval.

Record `lscpu`, CPU flags, OS, package versions, checkpoint SHA and runner load.
Pin package versions to the first run when comparing machines. The above Python
3.12 environment is easier to reproduce on ordinary runners than this host's 3.14;
record any dependency-version difference. Warm up 5 times, then measure 30 runs;
report median, p95, 1/4 threads, artifact size and memory. Repeat entire runs in
reversed candidate order before accepting a small gain. `--measure-only` reuses
graphs/inputs for repetitions. Linux RSS includes runtime, weights and working
buffers; it is not the model's isolated tensor-memory usage.

Web uses the same pinned ORT **1.20.1** as `web/public/worker.js`:

```sh
NPM_CONFIG_CACHE=/tmp/e04-npm npm install --prefix /tmp/e04-web onnxruntime-web@1.20.1 playwright
PLAYWRIGHT_BROWSERS_PATH=/tmp/e04-browsers /tmp/e04-web/node_modules/.bin/playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=/tmp/e04-browsers NODE_PATH=/tmp/e04-web/node_modules \
  node scripts/benchmark_fft_web.cjs /tmp/e04
```

The script binds only 127.0.0.1 and sets COOP/COEP for WASM threads. Each case gets
a fresh Chromium process; it checks output length and finite audio, and reports
numerical errors against the same dense CPU ONNX reference. Browser RSS is an
instantaneous sum over non-host browser processes after inference, **not peak RSS
or WASM heap size**; compare within this metric, not directly to native peak RSS.
The Web run does not test the phonemizer/audio UI, Safari, Firefox, WebGPU or WebNN.

Primary implementation references:
[ONNX DFT specification](https://onnx.ai/onnx/operators/onnx__DFT.html),
[ORT CPU FFT implementation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/signal/dft.cc),
[ORT Web providers](https://onnxruntime.ai/docs/tutorials/web/).

## Acceptance and next approaches

Do not promote either experiment simply because it uses an FFT or is smaller.
Require a repeatable end-to-end latency improvement on target CPU and WASM,
numerical acceptance above, unchanged durations/length and acceptable memory.
If a benefit is limited to one runtime, document that scope and keep a clear dense
fallback for the others; do not switch the default globally.

If both candidates remain slower on AVX2, investigate a staged export with a
purpose-built FFT WASM DSP module (and a native FFT implementation on CPU),
including JS/native boundary costs and overlap-add. That is a new implementation,
not a proven result of this branch. Otherwise retain only the experiments/report
and explain that E04 acceleration remains unachieved. Open/update a draft PR,
never merge and never claim an undemonstrated gain.

## Validation recorded for handoff

* Existing export tests: 2 passed.
* New candidate tests: 3 passed (both FFT paths, CPU ORT, opsets 17/20).
* Chromium 159.0.8154.6, ORT Web/WASM 1.20.1: all 12 model/thread
  combinations executed with finite audio and the expected length. Both FFT
  paths were slower. `web-preliminary.json` records the exploratory measurements.
  That older run compares to the Torch dense wrapper, so its whole-model SNR
  includes pre-existing cross-runtime differences; it is **not** the FFT parity SNR.
  Fresh runs of the checked-in script prepare a dense CPU ONNX reference instead.
* A sequential CPU rerun with corrected process memory is in `cpu-clean.json`.
  Graphs/inputs were reused from the exploratory synthetic run (24 tokens, M2,
  18,432 audio samples); its error/SNR fields likewise use the older Torch reference.
  Use the new candidate tests for strict dense-ONNX FFT parity.

These are historical handoff results. The subsequent AVX2 measurements and
expanded CPU/Web fidelity checks are recorded in [avx2/REPORT.md](avx2/REPORT.md).
No listening validation has been completed. Acceleration E04 remains unfinished;
this branch contains experiments, not an accepted production optimization.

Sequential rerun (milliseconds and peak process RSS in MiB):

| Graph | Threads | Median ms | p95 ms | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| filter-dense.onnx | 1 | 25.11 | 35.49 | 77.30 |
| filter-dense.onnx | 4 | 19.32 | 35.91 | 77.75 |
| filter-dft.onnx | 1 | 63.88 | 113.03 | 82.41 |
| filter-dft.onnx | 4 | 60.99 | 68.79 | 82.59 |
| filter-radix2.onnx | 1 | 235.33 | 424.75 | 83.70 |
| filter-radix2.onnx | 4 | 91.12 | 149.80 | 83.02 |
| full-dense.onnx | 1 | 78.41 | 101.95 | 111.02 |
| full-dense.onnx | 4 | 48.66 | 88.94 | 111.50 |
| full-dft.onnx | 1 | 98.55 | 140.18 | 111.66 |
| full-dft.onnx | 4 | 58.19 | 73.94 | 111.60 |
| full-radix2.onnx | 1 | 155.10 | 205.86 | 113.79 |
| full-radix2.onnx | 4 | 108.81 | 150.76 | 114.22 |
