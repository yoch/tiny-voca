# E04 research closeout

E04 is complete for research. PR #1 remains a draft. Dense is still the
default, and the opt-in compact Web worker remains an experimental path.
There is no further production qualification in scope.

| Path | Measured latency reduction | Scope |
| --- | ---: | --- |
| Fused native, English | 17.8–18.1% | CPU EPYC, one ORT thread, ~2 s, text to audio |
| Compact Web, English | 24.5–28.8% | Chromium, Firefox and WebKit/WPE on Linux, one ORT thread, ~2 s, text to audio |
| Compact Web, English | 16.8–32.3% | Same Linux engines, four ORT threads, ~2 s. Chromium p95 regressed in one order |
| Late native FFT, French | 25.9–28.0% | EPYC, one ORT thread, two phrases, pre-tokenized inference and DSP |
| Late native FFT, French | 12.2–18.3% | Same French probe, four ORT threads |

Standard ONNX DFT lost to dense because transform kernels and intermediate
tensor traffic dominate these short fixed FFT sizes. Expressing every radix-2
butterfly as graph operators made that worse. A packed real FFT with reusable
plans and frame-local scratch removed much of that cost. In Chromium the early
boundary at the cepstral heads and **original** oscillator reduced stage output
from 1.97 MB to 0.30 MB for ~2 s. The full-F0 cut failed waveform parity
because small phase differences accumulated. The French checkpoint's
frequency-dependent mixed excitation makes the later spectral boundary the
tested native option. It keeps the original mix in ONNX.

The Web evidence covers the Linux runner, not Apple Safari or a broad device
distribution. X86 without AVX2 and ARM64 were exercised functionally under
emulation. No human blind listening result was collected. Numerical Web parity
was at least 115.08 dB SNR. The final FR speed probe preserved durations,
shapes and finite samples, with at least 129.73 dB SNR.

## Final native French probe

The published FR checkpoint, existing E04-C dense and late graphs, and
generic x86-64 C build were reused. Same feeds and fixed common noise.
Five warmups and 30 runs per case and variant, repeated in reverse order,
CPU affinity 0–3. Timings start after phonemization, session creation and
noise preparation. The FFT timings include ORT output materialization,
three C filter calls and waveform addition.

| Audio | ORT threads | Dense median (ms) | Late FFT median (ms) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 2.45 s, FR_F1 | 1 | 41.71–42.62 | 30.54–30.67 | 26.8–28.0% |
| 4.78 s, FR_M1 | 1 | 87.43–87.73 | 64.57–64.76 | 25.9–26.4% |
| 2.45 s, FR_F1 | 4 | 21.97–22.39 | 18.32–18.82 | 15.9–16.6% |
| 4.78 s, FR_M1 | 4 | 43.28–45.18 | 36.93–38.02 | 12.2–18.3% |

Run the native FR parity command in [E04-C reproduction](../e04c/REPRODUCE.md)
first. Then run the command below with the same French fork on PYTHONPATH,
`TINY_VOCA_LANGUAGE=fr` and eSpeak NG on PATH.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0-3 "$PYTHON" scripts/e04c/benchmark_fr.py \
  --checkpoint /tmp/e04c/fr.pt \
  --checkpoint-sha256 61c00f50dffeda1cc8773243b4722a1d64b57a016db90ecd96477aae6225b3a9 \
  --graphs /tmp/e04c/fr --library /tmp/e04c/native-x86-64.so
```

[FR raw timings and hashes](fr-native.json.gz) · [Algorithmic comparisons](../e04b/COST.md) ·
[Web qualification evidence](../e04c/REPORT.md)
