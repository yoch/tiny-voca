# E04 AVX2 rerun and staged CPU investigation

**Decision: keep dense production export.** DFT and radix-2 regress on CPU and WASM in both run orders. The staged native FFT remains an experiment; no global switch, checkpoint change, retraining or merge.

## Runner and protocol

AMD EPYC 9V74 VM, 5 visible vCPUs, cgroup quota 4 CPU equivalents. AVX2 **and AVX-512** are exposed; PyTorch reports AVX512. No ISA was forced. These results do not isolate AVX2 or explain differences from the old i7. Python 3.12.14; Torch 2.15.0+cpu, NumPy 2.5.4, ONNX 1.23.2, ORT CPU 1.32.0 (same pinned package versions as the original run). Chromium 159.0.8154.6, Playwright 1.65.1, ORT Web/WASM 1.20.1.

The unchanged public trained checkpoint has SHA256 `ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70`. Its fingerprint matches dizione at the commit in requirements.txt. Public prebuilt phonemizer data were downloaded; no phonemizer training was performed. Source/data revisions, CPU flags, quota, versions and initial load are in [environment.json](environment.json); all generated graph/input hashes and validation tokens are in [manifest.json](manifest.json).

All benchmark cases ran sequentially, with 5 warmups and 30 timed inferences in a fresh worker/browser, then the complete model/thread order reversed. No benchmark overlapped another benchmark or tests. The VM is shared; host contention cannot be excluded. All individual timings are retained in the six JSON files beside this report. p95 uses linear interpolation at rank 0.95 × (n−1). Preparation, session creation, reference construction and extra validation inputs are outside timing.

Timed inputs are identical across backends: two filters over **48,128 samples** (2.005 s), and full trained synthesis of **24 deterministic phone IDs, M2, 49,664 samples** (2.069 s). The six extra full-model inputs validate both voices, lengths 1/5/40, speeds 0.8/1/1.25, pitch shifts −0.1/0/0.1, pitch ranges 0.7/1/1.3 and three actual sentences. Sentence timing is not included in these performance tables. All use the same deterministic noise.

## CPU and Web results

Each cell is **median / p95 ms**. Forward and reverse are independent runs of the same graphs/inputs. These runs use trained weights, unlike the old synthetic `cpu-clean.json`.

### CPU

| Workload | Threads | Order | Dense | DFT | Radix-2 |
| --- | ---: | --- | ---: | ---: | ---: |
| filter | 1 | forward | 7.25 / 8.71 | 20.31 / 21.20 | 53.81 / 55.58 |
| filter | 1 | reverse | 7.58 / 7.97 | 21.23 / 22.28 | 56.96 / 63.40 |
| filter | 4 | forward | 3.01 / 4.31 | 15.91 / 18.69 | 19.11 / 27.49 |
| filter | 4 | reverse | 3.12 / 4.30 | 15.79 / 18.31 | 18.60 / 25.57 |
| full | 1 | forward | 29.39 / 30.98 | 42.39 / 44.00 | 76.92 / 79.56 |
| full | 1 | reverse | 30.21 / 32.09 | 43.99 / 47.20 | 81.36 / 92.01 |
| full | 4 | forward | 15.57 / 21.87 | 28.47 / 32.31 | 30.75 / 43.30 |
| full | 4 | reverse | 17.98 / 24.24 | 29.81 / 34.64 | 31.58 / 38.54 |

### Chromium WASM

| Workload | Threads | Order | Dense | DFT | Radix-2 |
| --- | ---: | --- | ---: | ---: | ---: |
| filter | 1 | forward | 25.73 / 28.13 | 31.46 / 44.48 | 87.65 / 97.47 |
| filter | 1 | reverse | 26.38 / 27.04 | 30.20 / 31.98 | 87.35 / 96.82 |
| filter | 4 | forward | 13.58 / 18.52 | 26.61 / 34.99 | 28.84 / 42.02 |
| filter | 4 | reverse | 10.92 / 17.99 | 26.44 / 32.65 | 28.48 / 47.49 |
| full | 1 | forward | 97.11 / 105.68 | 117.30 / 124.64 | 159.53 / 183.30 |
| full | 1 | reverse | 99.45 / 101.89 | 104.93 / 108.82 | 161.88 / 188.49 |
| full | 4 | forward | 44.75 / 70.88 | 59.69 / 91.65 | 67.45 / 86.19 |
| full | 4 | reverse | 44.76 / 71.42 | 57.67 / 85.28 | 59.39 / 90.58 |

### Artifact size and memory

RSS ranges cover both thread counts and both orders. CPU is peak process VmHWM before extra validation; Web is instantaneous summed RSS of non-host Chromium processes after inference, including reference buffers. Neither is isolated tensor memory; the two metrics are not comparable to each other.

| Graph | ONNX bytes | CPU peak RSS MiB | Web process RSS MiB |
| --- | ---: | ---: | ---: |
| filter-dense | 2,646,764 | 70.38–70.96 | 343.33–362.05 |
| filter-dft | 67,749 | 74.41–74.80 | 337.14–352.30 |
| filter-radix2 | 261,159 | 74.95–75.29 | 339.45–354.55 |
| full-dense | 21,386,859 | 107.74–108.74 | 385.29–394.30 |
| full-dft | 18,808,022 | 112.22–113.08 | 379.18–389.75 |
| full-radix2 | 19,001,550 | 115.34–116.20 | 381.20–396.03 |

## Fidelity

CPU candidates compare to dense CPU ONNX. Web candidates compare to a separately prepared dense WASM reference at the same thread count; the reference browser closes before the measured browser starts. Web `cpu_reference` fields separately retain the pre-existing cross-runtime difference. That difference already affects dense and is not FFT error. `snr_db: null, exact: true` means identical arrays, not a failed SNR calculation.

| Backend scope | Minimum SNR dB | Maximum normalized peak error | All shapes/durations and thresholds |
| --- | ---: | ---: | --- |
| CPU DFT/radix-2 | 125.62 | 6.95e-07 | pass |
| WASM DFT/radix-2 | 125.66 | 7.56e-07 | pass |
| CPU native staged | 129.13 | 5.56e-07 | pass |

Filter acceptance also runs through ONNX CPU at both FFT sizes, immediately before/on/after hop and window boundaries, including 48,001 samples, identity, silence, zero response and edge/boundary impulses. Absolute/relative tolerance: 3e-6 / 2e-5. Full-model acceptance: normalized peak <2e-5, SNR >100 dB, identical duration arrays and output shape. CPU tests cover DFT and radix-2 at opsets 17/20. **6 tests pass**, including existing production export tests; see [tests.log](tests.log). All 24 timed Web cases and 72 additional Web full-model comparisons pass same-runtime parity. There was no listening study, audio UI, phonemizer UI, Safari/Firefox, WebGPU or WebNN validation.

## Better approach investigated: native FFT outside ONNX

`scripts/benchmark_fft_native.py` exports the existing neural computation, responses, pulse train, noise slice and durations into a separate ONNX graph, then applies NumPy native FFTs. It preserves the same padding, windows, responses, four-way overlap-add, normalization and crop. Inputs and windows are loaded into RAM before warmup. Timed full synthesis includes `session.run`, materialization of **2,005,328 output bytes**, the Python/native boundary, framing, both forward/inverse transforms and overlap-add. Output transfer is included in the neural-stage time; it is not measured as an isolated memcpy. NumPy DSP stays single-threaded; 1/4 threads controls the ONNX stage.

Fresh dense controls run in the same suite, then the entire suite is reversed. Each cell is median / p95 ms.

| Workload | Threads | Order | Dense | Native FFT | Native neural stage median | Native DSP median |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| filter | 1 | forward | 7.41 / 7.61 | 4.34 / 4.64 | 0.00 | 4.34 |
| filter | 1 | reverse | 7.51 / 8.30 | 4.33 / 4.71 | 0.00 | 4.32 |
| filter | 4 | forward | 3.12 / 3.44 | 4.36 / 4.51 | 0.00 | 4.35 |
| filter | 4 | reverse | 3.17 / 7.58 | 4.77 / 5.19 | 0.00 | 4.76 |
| full | 1 | forward | 29.97 / 31.14 | 25.35 / 27.02 | 22.15 | 3.19 |
| full | 1 | reverse | 29.82 / 30.90 | 24.81 / 26.36 | 21.77 | 3.01 |
| full | 4 | forward | 15.43 / 17.06 | 15.75 / 19.97 | 12.58 | 3.02 |
| full | 4 | reverse | 14.83 / 17.14 | 14.90 / 17.79 | 11.80 | 3.00 |

Full synthesis median latency reduction versus same-suite dense at 1 thread(s): **15.4% forward, 16.8% reverse** (negative means regression).

Full synthesis median latency reduction versus same-suite dense at 4 thread(s): **-2.1% forward, -0.5% reverse** (negative means regression).

The stage graph is 18,739,749 bytes plus 3,578 bytes of window data (excluding Python/runtime libraries). Full-stage peak process RSS is 109.98–110.97 MiB. This is a **CPU-only prototype**. There is no equivalent WASM module, JS/WASM transfer measurement or listening evaluation. The measured CPU result does not establish a Web benefit and is insufficient to change the global exporter. Keep it isolated; the next implementation should evaluate a purpose-built WASM FFT DSP boundary with the same acceptance protocol.

## Reproduce

From this branch, with `uv`, Node and the Chromium runtime libraries available:

```sh
uv venv --python 3.12 /tmp/e04-venv
uv pip install --python /tmp/e04-venv/bin/python -e . -r docs/e04/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
curl -fL https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/tiny-voca.pt -o /tmp/e04-checkpoint.pt
echo "ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70  /tmp/e04-checkpoint.pt" | sha256sum -c -
mkdir -p /tmp/e04-phonemizer
for file in LICENSE.cmudict LICENSE.homograph-data align.json g2p.json g2p.npz homographs.json lexicon.json lexicon.tsv sources.json; do
  curl -fL "https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer/$file" -o "/tmp/e04-phonemizer/$file"
done
export DIZIONE_DATA=/tmp/e04-phonemizer OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/e04-venv/bin/python -m pytest tests/test_export.py tests/test_fft_candidates.py -q
/tmp/e04-venv/bin/python scripts/benchmark_fft.py --checkpoint /tmp/e04-checkpoint.pt \
  --output /tmp/e04-avx2 --sentences --runs 30
/tmp/e04-venv/bin/python scripts/benchmark_fft.py --checkpoint /tmp/e04-checkpoint.pt \
  --output /tmp/e04-avx2 --measure-only --reverse --result-name cpu-reverse.json
npm install --prefix /tmp/e04-web onnxruntime-web@1.20.1 playwright@1.65.1
export PLAYWRIGHT_BROWSERS_PATH=/tmp/e04-browsers NODE_PATH=/tmp/e04-web/node_modules
/tmp/e04-web/node_modules/.bin/playwright install chromium
node scripts/benchmark_fft_web.cjs /tmp/e04-avx2
node scripts/benchmark_fft_web.cjs /tmp/e04-avx2 --reverse --result-name=web-reverse.json
/tmp/e04-venv/bin/python scripts/benchmark_fft_native.py --output /tmp/e04-avx2 \
  --checkpoint /tmp/e04-checkpoint.pt
/tmp/e04-venv/bin/python scripts/benchmark_fft_native.py --output /tmp/e04-avx2 \
  --measure-only --reverse
```

Use a fresh output directory to prepare a new dataset. `--measure-only` reuses exact graphs/noise/references. Raw results are written into the output directory; retain each repetition separately. Check the downloaded phonemizer hashes against environment.json. The initial package freeze is in [requirements-freeze.txt](requirements-freeze.txt).
