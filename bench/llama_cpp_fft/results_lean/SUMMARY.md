# tiny-voca host synth in llama.cpp: lean candidates, summary

Target: llama.cpp master `e300919cc704ddde3a8a8346a8714f57eeeee71b` (2026-11-22). `tiny-voca.cpp`, `mtmd-audio.cpp`
and `mtmd-audio.h` are identical to `9d6f04f`. Machine: 4-vCPU KVM VM, Intel Xeon @ 2.10GHz (Emerald Rapids class,
AVX-512), g++ 13.3, clang 18.1. Real compile line of `tiny-voca.cpp` (default Release build): `-O3 -DNDEBUG -fPIC`
plus the project warnings, no `-march`, inside a UNITY TU; `mtmd-audio.cpp` is a separate TU. Details, ablation and
method: `ABLATION.md`.

## Candidates (all only in `tools/mtmd/models/tiny-voca.cpp`)
| | what | code lines +/- | comment lines +/- | new structs / functions | special cases |
|---|---|---|---|---|---|
| **LEAN** (`lean/lean.patch`) | the 3 FFTs per STFT frame use a packed real FFT `tiny_voca_rfft` (n/2-point iterative radix-2 complex FFT, tables built once per filter call, first two stages fused, stack scratch) instead of `mtmd_audio_fft`/`mtmd_audio_ifft`; the rest of the filter keeps its buffers and loops | +83/-10 | +5/-1 | 1 struct (ctor + 3 methods) | the fused size-2/4 stage |
| **MID** (`lean/mid.patch`) | LEAN + the pulse train of variant C (one `exp` per frame, phase in cycles, two sin/cos pairs per sample), an independent hunk | +100/-23 | +6/-1 | same | same |
| **FULL** (`lean/full.patch`) | MID + filter response once per envelope (`cep` returns the gain, which is added to the log magnitude; FFT + cos/sin are skipped when the cepstrum did not change) | +118/-33 | +9/-3 | same | same |
| C (reference) | `tiny-voca-rfft-v2.patch` | +150/-55 | +11/-5 | 1 struct, 3 std::function callbacks, static plans | fused stage, normalization tables |

Each change can be read on its own: in LEAN only the FFT calls change (plus `rout[i]` for the real output and one
removed mirror loop). MID's extra hunk only touches `tiny_voca_pulse_train`. FFT plans are rebuilt on each filter
call (twice per synth): there is no static state. There is no change outside `tiny-voca.cpp`, no new file and no
new API.

## Synth time at the real flags
Real `libmtmd.so` of each build, called in-process (`lean/harness/real.cpp`), all 8 dumps (2868 frames) per
sample, 40 interleaved ABBA pairs, pinned core. Ratio = pair-ratio median [q25, q75]; ref is 465 ms per pass.

| | vs original | vs C | share of C's gain | -march=native (standalone harness) |
|---|---|---|---|---|
| LEAN | **0.310** [0.288, 0.314] | 1.33 [1.28, 1.36] | **90 %** | 0.302 |
| MID | **0.273** [0.270, 0.281] | 1.17 [1.12, 1.18] | **95 %** | 0.245 |
| FULL | **0.252** [0.244, 0.256] | 1.09 [1.07, 1.13] | **97 %** | 0.231 |
| C | 0.232 [0.219, 0.236] | 1 | 100 % | 0.226 |

Standalone harness at `-O3 -DNDEBUG -fPIC`, same workload: LEAN 0.302, MID 0.260, FULL 0.248, C 0.233 (the two
harnesses agree once the FFT scratch is on the stack, see ABLATION.md).

## llama-tts (`llama-tts.txt`)
llama-tts end to end, TV_TIMING, 5 builds interleaved with the order rotated every repetition, median of 10,
voices F2/M2, short/medium/long text, threads 1 (pinned) and 4. Ranges over the 12 configurations:

| | synth vs master | stacks + synth vs master | whole-process wall vs master |
|---|---|---|---|
| LEAN | 0.299 - 0.329 | 0.64 - 0.81 | 0.79 - 0.98 |
| MID | 0.256 - 0.327 | 0.61 - 0.83 | 0.82 - 0.96 |
| FULL | 0.231 - 0.267 | 0.59 - 0.82 | 0.77 - 1.01 |
| C | 0.211 - 0.268 | 0.59 - 0.80 | 0.79 - 0.98 |

Example, long F2 (1022 frames, 10.9 s of audio), 1 thread: synth 156.1 ms on master, 49.2 with LEAN, 43.4 with MID,
39.6 with FULL, 35.8 with C. With 4 threads: 158.7, 49.2, 42.9, 39.8, 36.5. Short texts are dominated by model
load (wall ~250 ms), so the wall ratio there is mostly noise.

## Parity and build
- Standalone harness, every dump plus 1-, 2- and 3-frame inputs, against the original synth: same length, worst SNR
  122.2 dB for LEAN and MID and 121.6 dB for FULL (C: 121.6), at most 1 LSB of 16-bit difference, 2 runs
  bit-identical. With `-march=native`: 120.5-120.7 dB, 1 LSB.
- llama-tts WAVs (8 utterances, F2/M2, 4 threads) against the master build: same length, at most 1 LSB, for every
  candidate. Two LEAN runs give identical WAVs. The RNG draws are unchanged.
- Clean build: 0 warnings in the real unity build (g++ 13), and 0 with g++ and clang++ 18 on the file alone at the
  project's warning flags (`raw/warnings.txt`). ASCII only, no tabs, no trailing whitespace, 4 spaces,
  `float * x` / `const T & a` style. C++17 with no intrinsics, attributes or fast-math. `git apply --check` is clean on
  `e300919cc`.

## Recommendation: LEAN, with MID's pulse-train hunk as an optional second commit
LEAN is one idea, "use a proper real FFT for the 3 transforms per STFT frame": one self-contained struct of about
80 lines and 8 changed lines in the filter. It gives 90 % of variant C's synth gain (synth 0.31x, about 3.2x
faster in llama-tts at 1 and 4 threads) with 93 code lines changed against C's 205, and it has the best gain per
changed line of any candidate that reaches 80 % (ABLATION.md). The FFT is where the time goes (76 % of the
original synth). The smaller options were measured and fall short:
- Reusing `mtmd_audio_fft` (half-size trick, tables once, response caching, pulse train, normalization) stops at 61 %
  of C's gain, because the recursive FFT is the bottleneck.
- A smaller FFT that reads its twiddles from the existing cache reaches 79 %.
- Dropping the fused first stages saves 8 lines and costs 15 % of synth time.

If maintainers accept a second small, independent hunk, MID (+17/-13 code lines in `tiny_voca_pulse_train` only)
brings the gain to 95 % of C. FULL's response caching adds another +18/-10 lines for 7 % more, and C's remaining
pieces (static plans, precomputed normalization, three `std::function` callbacks) add 50 lines for the last 8 %.
Neither is worth the review cost in a first PR.

Not proposed: a change to `mtmd_audio_fft` itself, i.e. making the shared recursive FFT iterative. tiny-voca is the only caller of the exported
`mtmd_audio_fft`/`mtmd_audio_ifft` outside `mtmd-audio.cpp`. But the same internal `fft`/`ifft` sit behind the mel
front ends of the ~15 audio preprocessors (`mtmd_audio_mel_energies`, `mtmd_audio_log_mel_frame`, kaldi fbank),
the zonos3 synth and one more inverse transform. Changing it would speed those up too, but it changes their
numerics and needs their own parity checks, so it would be a separate PR with its own measurements on those
models. It was not built or measured here, and LEAN does not need it.

## Deviations
- No candidate touches `mtmd-audio.cpp`. The separate outside-file candidate was argued above but not built.
- The ablation table uses the u5+u6 workload (medium F2 + M2, 557 frames) per sample. The final candidates use
  all 8 dumps.
- The standalone harness alone was misleading for the first candidates (member-vector scratch is about 20 % slower
  in the unity build than in a standalone TU). So every claim here is measured on the real `libmtmd.so` of each
  build, rebuilt in the default unity build, and the standalone harness is kept for parity and for the
  `-march=native` column. The first llama-tts run with the member-vector versions is kept as
  `raw/tts_run1_memberscratch_raw.txt`.
- The `-march=native` column comes from the standalone harness, not from a native llama.cpp build.
- The WAV parity reference is master plus `timing-instrumentation.patch`, which only adds timers in `mtmd.cpp`.
- The kit's `run.sh` was not run as is: its fixed commit, its `tts-threads` patch and its `-DGGML_NATIVE=ON` do not
  apply to master or the real build. `lean/tts/run_tts.sh` keeps its llama-tts loop (texts, voices, seed,
  TV_TIMING, medians of 10, WAV parity) for N builds with a rotating order, pins the 1-thread runs, and adds wall
  time.
- One machine only: a 4-vCPU cloud VM, so drift and noisy neighbours are possible. Every ratio comes from
  interleaved pairs, and the quartiles are given.
