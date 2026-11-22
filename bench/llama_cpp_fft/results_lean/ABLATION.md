# Ablation: tiny-voca host synth in llama.cpp, at the real build flags

## Setup
- Machine: cloud VM, Intel Xeon @ 2.10GHz (family 6 model 207, Emerald Rapids class), 4 vCPU (KVM),
  AVX-512 available, 15 GiB RAM, Ubuntu 24.04, g++ 13.3.0, clang++ 18.1.3, cmake 3.28.3 (`raw/host.txt`).
- llama.cpp master `e300919cc704ddde3a8a8346a8714f57eeeee71b` (2026-11-22). `tiny-voca.cpp`, `mtmd-audio.cpp`
  and `mtmd-audio.h` are byte-identical to `9d6f04f`, and the tiny-voca part of `models.h` is unchanged
  (`raw/llama_commit.txt`). `tts-threads-9d6f04f.patch` was not applied (PR 32664 is in master, `-t` already reaches
  mtmd). `timing-instrumentation.patch` was applied to every llama-tts build for TV_TIMING (it only touches
  `mtmd.cpp`) and is not part of any candidate.
- Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release` with default options. Real compile line of the target that holds
  `tools/mtmd/models/tiny-voca.cpp` (`build/tools/mtmd/CMakeFiles/mtmd.dir/flags.make`, `raw/flags.make.mtmd.txt`):

      -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual
      -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-cast-qual -Wno-maybe-uninitialized  (-std=gnu++17)

  So the expectation is confirmed: `-O3 -DNDEBUG`, no `-march` (GGML_NATIVE=ON only reaches the ggml CPU
  backend). Two details matter for performance: `-fPIC` (libmtmd is a shared library, so out-of-line inline
  functions are exported weak symbols), and the UNITY build. `tiny-voca.cpp` is compiled inside
  `Unity/unity_3_cxx.cxx` with 15 other model files, while `mtmd-audio.cpp` is a core source excluded from unity,
  so `mtmd_audio_fft` is never inlined.
- Measurement, two harnesses, both A/B in one process, pinned to core 2, ABBA order, ratio of medians plus the
  quartiles of the per-pair ratios:
  - **real**: `lean/harness/real.cpp` dlopens two copies of `build/bin` and calls the exported
    `mtmd_graph_tiny_voca::synth` of each `libmtmd.so`, so this is exactly the code llama-tts runs. Every variant
    is rebuilt in the real unity build (`lean/harness/rebuild_real.sh`, about 7 s each).
  - **standalone**: `lean/harness/` (each candidate's synth section in its own `-O3 -DNDEBUG -fPIC` TU, the
    mtmd FFT in another TU, one shared lib). Used for parity and for the `-march=native` column.
- Workloads: the ablation uses `u5+u6` (293 + 264 frames, medium F2/M2) per sample, 40 pairs. The final candidates
  use all 8 dumps (2868 frames) per sample, 40 pairs.

### Why the real-library harness
The first round of candidates kept the FFT scratch in member `std::vector`s. In the standalone harness they measured
0.294 (lean + response caching), but in llama-tts they gave 0.35. Callgrind on the real `libmtmd.so` showed why: in
the unity TU, GCC inlines `tiny_voca_filter` into `synth` and leaves `tiny_voca_rfft::forward` out of line (a
weak exported symbol). Out of line, the stores to the output may alias the scratch held in `this`, so the loops
are not vectorized (188 M instructions on u5 against 130 M for the same source in the standalone harness). The
fix is the one variant C already uses: `const` methods with `float r[256], im[256]` scratch on the stack. With it,
the real build and the standalone harness agree to within 1-5 %, and lean's synth goes from 0.37 to 0.31. Every
number below comes from the real build unless marked otherwise.

## Profile of the original synth (standalone harness, all 8 dumps, ~560 ms per pass)
| part | share |
|---|---|
| 2 forward `mtmd_audio_fft` per STFT frame (response + source) | 50 % |
| inverse `mtmd_audio_ifft` per STFT frame | 25 % |
| pulse train (exp, fmod, 3 sin/cos per sample, double) | 10 % |
| response exp/cos/sin per bin | 6 % |
| cepstrum fill, windowing, product, mirror, overlap-add, normalization | 3 % |
| noise RNG | 2 % |

The recursive `mtmd_audio_fft` is three quarters of the time. Microbenchmark (`lean/harness` style, 512 points):
`mtmd_audio_fft` on 512 real samples takes 9.9-12.6 us, `mtmd_audio_ifft` on 256 complex values 4.5 us. The
iterative 256-point complex FFT of variant C takes 0.58-0.75 us, and its whole 512-point real FFT 0.8-0.9 us. With
twiddles read from `mtmd_audio_cache::sin_vals` at stride n/len, the same iterative loop takes 2.1-2.2 us: the
strided twiddle loads block vectorization. The first two fused stages are worth another 1.4x.

## Ideas (each alone on the master file, then combined)
| id | idea | where |
|---|---|---|
| Pw | phase kept in cycles in [0, 1) with `floor`, no `fmod` of an ever-growing phase | pulse train |
| Pt | `cos((whole + 1) theta)` from the two angles already needed (2 sin/cos pairs) | pulse train |
| Pc | variant C's pulse train: Pw + Pt + one `exp` per frame (geometric F0 ramp) | pulse train |
| T | sin/cos table and Hann window built once (static) instead of per filter call | filter |
| N | overlap-add normalization precomputed (squared windows are hop-periodic), no `env` buffer | filter |
| R | response once per envelope: `cep` returns the gain, the filter skips FFT + cos/sin when the cepstrum did not change, the gain is added to the log magnitude | filter + synth lambdas |
| H | real FFT as an n/2-point complex FFT with the existing `mtmd_audio_ifft` (conjugation trick + split step, twiddles from the existing cache) | filter, reuse |
| I | H with a 25-line in-file iterative radix-2 FFT on interleaved data, twiddles from the cache table | filter |
| F | the FFT of variant C, trimmed: `tiny_voca_rfft` with a bit-reversal table, contiguous per-stage twiddles and post-rotation tables, stack scratch, built per filter call (no static plans), first two stages fused; the original buffers and loops are kept | filter |

## Table (real build, `-O3 -DNDEBUG -fPIC`, u5+u6, 40 ABBA pairs)
Ratio = synth time / original (pair-ratio median, [q25, q75]). Gain = 1 - ratio. "of C" = gain / gain of C.
Lines = changed lines vs the master file as git counts them: code (+/-), comment-only (+/-); blank lines are
not counted (they are all additions, at most +5).

| variant | ratio [q25, q75] | gain | of C | code +/- | comment +/- | gain per 10 code lines |
|---|---|---|---|---|---|---|
| Pw | 0.988 [0.970, 1.015] | 1 % | 2 % | +3/-2 | 0/0 | (noise) |
| Pt | 0.995 [0.928, 1.023] | 1 % | 1 % | +4/-3 | +1/0 | (noise) |
| Pc | 0.957 [0.941, 0.984] | 4 % | 6 % | +17/-13 | +1/0 | 1.4 % |
| T | 0.997 [0.975, 1.015] | 0 % | 0 % | +9/-3 | +1/0 | 0 |
| N | 0.991 [0.959, 1.011] | 1 % | 1 % | +9/-5 | +1/0 | (noise) |
| R | 0.894 [0.882, 0.923] | 11 % | 14 % | +20/-12 | +2/-1 | 3.3 % |
| H (reuse mtmd FFT) | 0.633 [0.619, 0.663] | 37 % | 47 % | +38/-15 | +5/-1 | 6.9 % |
| H+R | 0.584 [0.573, 0.592] | 42 % | 53 % | +53/-22 | +7/-2 | 5.5 % |
| H+R+Pc | 0.550 [0.532, 0.571] | 45 % | 58 % | +70/-35 | +8/-2 | 4.3 % |
| H+R+Pc+N (best reuse-only) | 0.524 [0.516, 0.533] | 48 % | 61 % | +79/-40 | +9/-2 | 4.0 % |
| I+R+Pc+N (small FFT, cache twiddles) | 0.389 [0.381, 0.398] | 61 % | 79 % | +109/-40 | +11/-2 | 4.1 % |
| F without fused stages | 0.378 [0.374, 0.385] | 62 % | 80 % | +75/-10 | +4/-1 | 7.4 % |
| **LEAN** = F | **0.314 [0.312, 0.317]** | **69 %** | **88 %** | **+83/-10** | **+5/-1** | **7.4 %** |
| LEAN + R | 0.296 [0.291, 0.298] | 70 % | 90 % | +101/-20 | +8/-3 | 5.8 % |
| **MID** = LEAN + Pc | **0.271 [0.267, 0.274]** | **73 %** | **94 %** | **+100/-23** | **+6/-1** | **5.9 %** |
| **FULL** = LEAN + Pc + R | **0.254 [0.248, 0.257]** | **75 %** | **96 %** | **+118/-33** | **+9/-3** | **4.9 %** |
| FULL + N | 0.245 [0.239, 0.250] | 76 % | 97 % | +127/-38 | +10/-3 | 4.6 % |
| C (tiny-voca-rfft-v2.patch) | 0.222 [0.214, 0.227] | 78 % | 100 % | +150/-55 | +11/-5 | 3.8 % |
| C without fused stages | 0.272 [0.264, 0.276] | 73 % | 94 % | +142/-55 | +10/-5 | |
| C with the original pulse train | 0.263 [0.258, 0.267] | 74 % | 95 % | +133/-42 | +10/-5 | |
| C with the original normalization | 0.230 [0.224, 0.238] | 77 % | 99 % | +143/-52 | +10/-5 | |

Steps, measured directly as pairs (B / A, real build, u5+u6, 40 pairs):

| step | ratio [q25, q75] |
|---|---|
| F without fusion -> LEAN (fuse the first two stages, +8 code lines) | 0.853 [0.833, 0.865] |
| LEAN -> MID (+ Pc) | 0.863 [0.852, 0.884] |
| LEAN -> LEAN + R | 0.941 [0.937, 0.949] |
| MID -> FULL (+ R) | 0.933 [0.928, 0.937] |
| FULL -> FULL + N | 0.985 [0.965, 1.008] |
| FULL -> C | 0.918 [0.904, 0.925] |
| best reuse-only (H+R+Pc+N) -> FULL | 0.470 [0.453, 0.479] |

Raw: `raw/real_ablation_u5u6.txt` (see the naming note on its first line), standalone harness:
`raw/harness_ablation_u5u6.txt`.

Reading:
- The FFT is the change that matters. Reusing `mtmd_audio_fft` at half size (H, 54 code lines) is the best gain
  per line of the reuse family, but it stops at 47 % of C's gain. All reuse ideas together (H+R+Pc+N) reach 61 %,
  because the recursive FFT copies at every level and its twiddles are strided. Unlike the audio.cpp exercise,
  where the framework FFT was already iterative, here the existing FFT is the bottleneck, so a lean patch has to
  bring its own FFT.
- The FFT has to be the fast kind: contiguous per-stage twiddles plus the two fused first stages. A smaller FFT
  that reads its twiddles from the existing cache (I) loses 0.39 against 0.31. Dropping the fused stages saves
  8 code lines and costs 15 % of synth time. Stack scratch in `const` methods is required in the unity build
  (see above).
- Once the FFT is fast, the pulse train (Pc) is the next best step (-14 %), ahead of response caching (R, -6 %),
  and both stay worthwhile. T and N are noise-level and are left out.
- New simple ideas measured and not kept: tables once (T: nothing, the tables are built twice per synth call);
  `cbuf` comparison as the cache key for R (simpler than C's three `std::function`s; kept in FULL); a smaller FFT on
  interleaved data with cache twiddles, twiddle-hoisted loop order J (I: 0.47 -> J: 0.44 alone, standalone harness);
  member-vector scratch (slower in the unity build, see above); rotation table of size m instead of m + 1
  (rot[m] is never read; kept).

## -march=native (secondary, standalone harness, all 8 dumps, 40 pairs)
| candidate | -O3 -DNDEBUG -fPIC | + -march=native |
|---|---|---|
| LEAN | 0.302 | 0.302 |
| MID | 0.260 | 0.245 |
| FULL | 0.248 | 0.231 |
| C | 0.233 | 0.226 |

(The original synth itself gets about 4 % faster with -march=native: 442 -> 428 ms per pass.)

## Parity (standalone harness, every dump + 1, 2, 3-frame cuts of u1, vs the original synth)
| candidate | -O3 worst SNR | worst 16-bit diff | same length | deterministic | -march=native worst SNR / LSB |
|---|---|---|---|---|---|
| LEAN | 122.2 dB | 1 LSB | all | yes (2 runs identical) | 120.7 dB / 1 |
| MID | 122.2 dB | 1 LSB | all | yes | 120.7 dB / 1 |
| FULL | 121.6 dB | 1 LSB | all | yes | 120.5 dB / 1 |
| C | 121.6 dB | 1 LSB | all | yes | 120.5 dB / 1 |

All pass the bar (SNR >= 110 dB, <= 1 LSB). Per-utterance numbers: `raw/harness_parity_O3.txt`,
`raw/harness_parity_native.txt`. The RNG draws are unchanged (same `normal_distribution` loop, same order).
