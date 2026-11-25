# Cost of Tiny Voca synthesis

The measured path is batch-one, FP32 ONNX, with 24 kHz output and an acoustic
hop of 256. The acoustic network has seven frame-level transformer blocks
(two prosody, five acoustic), width 192, four heads and feedforward width 640.
Four more blocks run at token length. Analysis below uses the existing
24-token, 194-frame, 49,664-sample E04 benchmark for the CPU operator trace.
The end-to-end experiment uses five real texts.

## Dataflow and arithmetic

One multiply-add is counted as two FLOPs. These are operation counts, independent
of a backend's padding, packing, fused instructions and special cases.

| Stage | Cost at 194 acoustic frames |
| --- | --- |
| Encoder, duration, prosody and acoustic stacks | About 1.35 GFLOP in transformer projections and attention alone. Each block at length L costs L(8d² + 4d×640) + 4L²d, d=192. Duration CNN, rotary tables, depthwise conv, softmax, masks and activations are extra. Attention grows quadratically per utterance. |
| Two coefficients heads | 98 harmonic + 36 noise coefficients, about 10 MFLOP. Includes harmonic payload 80 causal + 16 anticausal + 2 gains and noise 32 causal + 4 gains. One additional F0 per acoustic frame. |
| Complex spectral responses | 391 harmonic frames × 257 bins, 779 noise frames × 129 bins. Two dense projections from 96/32 coefficients cost 51.45 MFLOP total. An exponential, sine and cosine per bin follow. Arrays of real and imaginary response alone account for 1.608 MB. |
| Harmonic source | 49,664 samples. Log F0 interpolation, an exponential, three trig evaluations and a square root per sample in the generic path. FP32 cumulative phase inside each acoustic frame, with a 24-bit fractional-cycle carry across frames. |
| Noise source | Fixed, identical random FP32 stream for comparison. ONNX slices to S samples, 198,656 bytes here. Random generation is excluded in all benchmark variants. |
| Windowing and framing | 399,616 windowed positions across the two branches (391×512 + 779×256). An explicit int64 gather grid can occupy 3.20 MB, before gathering and window multiplication. Dense graph incorporates analysis windows into convolution kernels. |
| Dense analysis and synthesis | Approx. 618.2 MFLOP from both real/imaginary analysis and inverse projection, including the separate window-square envelope convolution. Uses large regular contractions but creates intermediate channel-major arrays. |
| Fused real analysis and synthesis FFT | 14.99 MFLOP for complex butterflies alone, using packed N/2 transforms, plus real packing and reconstruction. Bit reversal and twiddles computed once. Two plans, window data and one-frame scratch fit in L1 on this host. |
| Spectral multiplication | About 1.21 MFLOP from six float operations for each of the 200,978 complex bins. Endpoints are real in the inverse. |
| Overlap-add, normalization and crop | Up to four overlapping Hann windows, about 400,000 windowed contributions before crop. Fused version accumulates directly in the cropped output and uses a precomputed reciprocal envelope. The first hop uses three windows. |
| Allocation and copy | Dense graph has dynamic output and intermediate buffers; the late stage makes two source waves and four response planes visible to the host. Fused DSP uses fixed-size stack scratch, two precomputed plans and reusable output buffers. There is no code-level allocation inside its C sample/frame loops. |

For T acoustic frames and Q tokens, outputs including int64 durations are:

| Boundary | Exact bytes | At T=194, Q=24 |
| --- | ---: | ---: |
| Dense audio | 1024T + 8Q | 198,848 |
| Late, two sources and four spectral arrays | 10288T + 9264 + 8Q | 2,005,328 |
| Compact, harmonic pulse samples plus coefficient heads | 1560T + 8Q | 302,832 |
| Early, F0 plus coefficient heads | 540T + 8Q | 104,952 |

The compact boundary reduces late output bytes by about 85%. The early boundary
reduces them by about 95%. On Web, the compact backend preloads the deterministic
noise stream once into the DSP heap. Requests copy neural outputs from ORT to
that heap and copy the audio back. ORT-to-JS output materialization is included
in ORT time and cannot be separately timed using the stock API.

## Operator-level evidence and roofline

The CPU ORT profile reports the mean of five profiled runs after three warmups.
These times include instrumentation overhead and vary with thread scheduling.
The archived node list records invocation counts and logical activation,
output and parameter bytes. Logical sizes double-count some buffers and views.
They measure tensor exposure, with neither allocation counts nor actual DRAM
traffic inferred from them. Web's stock ORT interface exposes an end-profiling
call that discards the WASM profile filename. Web figures come from measured
graph boundaries and engine/copy phases in real Chromium.

| Profile group | ORT 1 thread (ms) | ORT 4 threads (ms) |
| --- | ---: | ---: |
| Dense graph, total kernel time | 33.66 | 37.80 |
| Dense, modules by neural scope | 21.55 | 27.84 |
| Dense analysis convolutions | 3.44 | 1.78 |
| Dense synthesis and envelope convolutions | 3.90 | 2.54 |
| Dense four cepstral projections | 0.54 | 0.29 |
| Dense spectral exponential/trig | 0.80 | 0.93 |
| Dense selected oscillator kernels (phase scans, trig, log, exp, sqrt) | 0.51 | 0.56 |
| Dense external noise gather | 0.01 | 0.02 |
| DFT, four transform kernels | 13.14 | 21.08 |
| DFT, four largest framing/reconstruction gathers | 6.53 | 5.85 |
| Radix-2 graph, total kernel time | 93.31 | 43.62 |

The four-thread profiled events contain scheduling outliers, including 3.5 ms
for one neural Slice. Use the warm unprofiled tables to compare latency. The
analysis convolutions nearly halve their profiled time with four threads;
dense transposed projections benefit less. ORT CPU uses MLAS and the worker
pool for these contractions. ORT's DFT handles the short power-of-two sizes
via a general radix-2 implementation and generic strided complex data. The
published source iterates over independent slices and its radix-2 routine
does not use an intra-op pool. Bluestein is for other sizes. Our graph also
creates a full Hermitian inverse, windows and overlap-add nodes. The measured
cost includes those nodes separately. Radix-2 ONNX materializes slices, mul,
add and concat tensors at each stage. Its summed logical outputs are 442 MB
per call, against 154 MB for DFT and 114 MB for dense. This is consistent with
materialization and dispatch dominating graph butterflies. It does not
identify hardware stall cycles.

An empirical single-core roofline uses median throughput of 30 FP32 1024²
matrix multiplications (about 104 GFLOP/s) and copies of a 128 MiB array
(about 37.5 GB/s, counting reads and writes). The crossover is about 2.8
FLOP/byte. These are local proxies, not hardware maxima. Dense filter
arithmetic alone would take 5.9 ms at that matrix throughput, close to
7.34 ms of profiled convolutions. In the fused case the butterflies' compute
floor is about 0.14 ms at the same throughput, but short staged FFTs do not
reach matrix-kernel utilization. The lower bound to move the 2.0 MB late
payload once is 0.05 ms of wall time at 37.5 GB/s (0.11 ms for one read plus
one write). The measured JS/WASM copies and materialization are separate.
Making the boundary smaller also moves the spectral response arithmetic,
so a raw-byte roofline alone cannot predict the end-to-end winner.

At one thread, removing every dense convolution tick from this profile would
remove 7.34/33.66 = 21.8% of CPU time (ideal speedup 1.28× by Amdahl). Even
a DSP kernel made twice as fast would remove at most 10.9% in that model.
Adding the projections and trig to the zero-cost region raises the bound to
25.8%, before any extra FFT/transfer cost. At four threads the profiled filter
share is about 11% under scheduling outliers. Workload-specific estimates based
on unprofiled graph-boundary measurements are in the benchmark report. The
network's quadratic attention reduces the filter share on longer texts.

Sources: [ORT DFT kernel](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/signal/dft.cc),
[ORT ConvTranspose](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/conv_transpose.cc),
[ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

## Approaches compared

| Approach | Small-transform fit and maintenance | Decision |
| --- | --- | --- |
| NumPy pocketfft | BSD-3-Clause backend. FP32 real batched transforms, good control. Existing path materializes framed and windowed batches and an overlap envelope, and crosses Python on each utterance. | Measured CPU control. Keep as reference. |
| PFFFT | Permissive FFTPACK-derived terms. Fixed real sizes with reusable plans and scratch, SIMD builds for WebAssembly, SSE, and NEON. Its internal order can permit spectral multiplication in place. A library swap still has to preserve the nonuniform response, padding, and overlap normalization. | Strong future drop-in for the shared core. License, supported sizes and SIMD reviewed. No performance number without building and measuring that combination. |
| KissFFT | BSD-3-Clause, real transforms and compact portable C. Optional SIMD support varies with build. | Useful simple reference. Chose the shared packed-real implementation to fuse framing and overlap-add around the transform. |
| FFTW | Planner and batched/small transforms could be effective, but GPL or alternative commercial licensing adds distribution terms to an Apache-2.0 project. | Excluded from this experiment. |
| Fixed C packed-real FFT | Two precomputed plans, contiguous butterfly lanes, no frame-batch allocations. Same DSP in native and WASM. Size-specific code and tests are the maintenance cost. | Measured at both retained graph boundaries. |
| Exact time-domain filter | Each frame needs a generally full N-tap circular FIR. The exponentiated cepstrum has no finite support guarantee; causal/anticausal factors do not imply a cheap, fixed FIR. Direct time-domain filtering would return to quadratic work. | No implementation. Truncation would require independent listening and fidelity validation. |
| Interpolation and reuse | Harmonic subframes are either the same envelope or the arithmetic mean of adjacent frame coefficients. Noise uses one envelope and four gains. Reuse the exact noise phase and log response, add each gain before clamping. Interpolating wrapped phase or complex responses after exponentiation changes the DSP. | Exact reuse implemented in the compact candidate. |
| Dense SIMDs and hybrid | Combining real/imag analysis into one convolution could share input packing, but it keeps dense inverse projection. A branch-specific hybrid can use dense where its threading wins and FFT elsewhere. | Dense remains the baseline and fallback. No evidence from this run supports a further hybrid. |
| Parallel DSP | Branches are independent; overlapped frames write shared samples. Parallelizing frames needs per-worker buffers or a reduction. Extra workers would contend with ORT's thread pool, especially on Web. | ORT uses 1/4 threads, each DSP invocation uses one. Scale and latency tradeoffs favor measuring a pooled variant only after choosing a boundary. |

Library references: [pocketfft](https://github.com/mreineck/pocketfft),
[PFFFT](https://github.com/marton78/pffft), [KissFFT](https://github.com/mborgerding/kissfft),
[FFTW license](https://www.fftw.org/faq/section1.html).
