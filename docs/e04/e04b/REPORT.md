# E04-B shared native and WASM vocoder

For about two seconds of speech, the fused CPU prototype reduces total median
latency by about **18% at one thread**, and the compact Chromium WASM prototype
by **25.5–28.1% at one thread** and **19.3–24.3% at four threads**. The values
span both run orders. Compact also reduces the 2 s ONNX boundary from 1,974,544
to 298,232 bytes. Both shared-DSP candidates pass same-runtime numerical parity.

The experiment compares two DSP boundaries using one packed-real FFT core. It
keeps the published checkpoint and the production exporter. The draft branch
contains the implementation, benchmarks, diagnostics and reference WAV workflow.

[COST.md](COST.md) gives the operation and byte model, the CPU operator trace
interpretation, a roofline and Amdahl bounds, and the library and algorithm
tradeoffs. [RESULTS.md](RESULTS.md) gives the Pareto comparison, per-stage
timings, copy sizes and fidelity. [LATENCY.md](LATENCY.md) gives medians, p95,
interquartile ranges and bootstrap intervals for both orders.

## Candidates

- **Dense** uses the original ONNX exporter and serves as the explicit fallback.
- **NumPy** is the previous CPU control with pocketfft, late source/response
  tensors, batch framing and overlap-add in Python/NumPy.
- **Fused** replaces the NumPy DSP with fixed real transforms in C. Windowing
  and bit reversal share the input pass. The forward DFT, spectral product,
  inverse and overlap-add reuse one frame's scratch. Overlap-add writes only
  into the retained output, with a separate first-hop normalization for the
  three windows that reach the start of the production crop.
- **Compact** runs the same filters from the two linear heads and the ONNX
  harmonic source. Sparse cepstra use a real FFT. The noise branch reuses the
  phase for four subframes with separate gains. DC gain is applied before the
  spectral clamp and exponential. The C code retains full filter responses
  and the same oscillator waveform, and moves approximately 85% fewer
  bytes across the boundary than fused.
- **F0-only diagnostic** sends only 135 floats per frame, but accumulated
  rounding and runtime-specific libm operations in the oscillator move its phase
  outside the acceptance threshold. Both runtimes reject it. It is retained as
  an explicit diagnostic switch, outside the acceptance timing matrix.

The two architecture candidates use the same source for GCC and Zig/Clang.
The native build targets x86-64-v3 by default. The WASM build uses SIMD128 and
does not need WASI imports in Chromium. The generated assembly contains
`v128.load` and `f32x4.mul`. The same source cross-compiles to an AArch64
object. On CPU the optional parallel probe
reuses a second DSP worker for the noise branch. Browser ORT can use a
four-thread pool; browser DSP runs synchronously in one instance.

The isolated fixed-core ISA comparison used the same trained 194-frame
stage outputs and 30 invocations in each direction. Baseline x86-64
measured 2.15–2.19 ms, AVX2 (x86-64-v3) 2.03–2.04 ms, and the AVX-512
build with 512-bit vectors 2.28–2.40 ms. The AVX-512 binary contains zmm
instructions. Wider vectors under this compiler and short transform mix
did not improve throughput. These are same-host binary choices, with no
conclusion about an older processor. [ISA timings](isa.json)

## Benchmark scope

The public checkpoint is verified by SHA256. The harness runs five real English
texts with expected lengths of 0.50, 1.00, 2.04, 5.02 and 10.12 seconds. It covers
the two voices and distinct speed, pitch shift and pitch range settings. The
[cases file](cases.json) records the exact text and inputs. All paths use
the same deterministic noise source; its generation is outside the timing.
Dense and fused receive only the samples required by the utterance.

The EPYC 9V74 VM exposes AVX2 and AVX-512. The affinity mask uses CPUs 0–3
with a four-CPU cgroup quota. All conditions run sequentially in two opposite
orders, with five warmups and 30 timings per case. Each variant/thread
combination starts a fresh process or browser. Raw JSON contains the timing
for every execution. The CPU native shared library uses the same ORT graph
as the corresponding Chromium candidate. Chromium uses ORT Web/WASM 1.20.1
and the checkpoint-matched dizione package through Pyodide 0.26.4, as in the
site worker. The CPU uses ORT 1.32.0 and native Python dizione.

The timer starts from text and ends with a normalized waveform buffer.
It includes phonemization, feed creation, inference, DSP, transfers and the
audio copies needed for delivery. Each case is one utterance, so first audio
becomes available when its synthesis completes. It does not measure browser
audio-device scheduling, DOM events, multi-sentence playback or sample
streaming within an utterance. Cold preparation measures model/runtime loading
on disk or localhost and a separate first request. It excludes remote network
variability and process launch. CPU RSS includes the Python frontend process
and imported Torch, and is sampled at the kernel high-water mark. Browser RSS
is the maximum sum observed at 100 ms intervals across Chromium processes,
including Pyodide and ORT, and can miss shorter peaks.

The compact graph plus WASM module totals 18,518,931 bytes versus 21,386,859
bytes for dense (13.4% fewer model/module bytes before transport compression).
The late WASM heap reserves 822 pages, compact 469, including a 16 MiB toolchain
stack reservation. The smaller boundary does not guarantee lower total browser
RSS. Pyodide plus model preparation took about 3.0–3.3 seconds on localhost;
the first cold utterance is more variable than steady-state inference.

## Fidelity and release decision

The acceptance comparisons use a dense reference from the same runtime and
thread count, with duration arrays included. Dense CPU versus dense WASM is
53.0–78.4 dB over these cases and is reported separately from FFT error.
Boundary checks exercise both window sizes with noise,
silence, identity, impulses, sinusoids and nonaligned lengths. The F0 diagnostic
also exercises constant pitch and transitions. Six existing tests and 80
additional filter comparisons passed. Short excerpts for both voices and
runtimes are in [audio/](audio). These PCM16 pairs share the same gain. The
reproduction scripts export the full float WAVs used for objective checks.
The objective checks and WAV generation do not constitute a human listening study.

The backend stays experimental. The production exporter and browser worker
continue to select dense. A rollout decision would require a human listening
comparison and broader browser and processor coverage. ARM64 and hardware
without AVX2 were unavailable for end-to-end measurement.

For a later integration, the compact boundary is the strongest Web candidate.
The one-thread gain holds at every measured length. At four threads, it holds
in both orders for 1–10 s, while the 0.5 s result changes direction. On CPU,
fused has stable one-thread gains and beats NumPy DSP time by about one third
at two seconds. At four threads, total gains are smaller and less stable.
The two-branch worker did not reliably reduce total CPU latency or even the
two-second DSP time. Keep the synchronous DSP and dense fallback for any
pilot. The timing data do not support a separate parallel DSP backend.

The F0-only boundary should be dropped from release consideration. Preserving
the ONNX oscillator is more important to waveform parity than the last 194 kB
of boundary savings for the two-second case. If further browser work is
needed, measure a smaller WASM stack and limit the audio scratch reservation
before adding a DSP thread pool. Those changes would address measured artifact
and memory costs without changing the acoustic definition.

See [REPRODUCE.md](REPRODUCE.md) for the commands, WAV output and data provenance.

Raw evidence: [CPU forward](cpu-forward.json.gz), [CPU reverse](cpu-reverse.json.gz),
[Web forward](web-forward.json.gz), [Web reverse](web-reverse.json.gz),
[parallel forward](parallel-forward.json.gz), [parallel reverse](parallel-reverse.json.gz),
[CPU operator profile](cpu-profile.json.gz), [roofline measurements](roofline.json),
[summary statistics](summary.json.gz), [environment](environment.json),
[CPU/WASM baseline parity](dense-cross-runtime.json),
[early CPU parity](cpu-early-parity.json), [early Web parity](web-early-parity.json),
[export tests](tests.log), [DSP checks](verify.log).
