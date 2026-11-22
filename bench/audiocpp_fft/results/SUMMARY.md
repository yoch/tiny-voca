# Summary: tiny-voca waveform head, second optimization pass

Machine: Intel Xeon @ 2.10 GHz (family 6 model 207), 4 vCPU KVM/Firecracker, AVX2 + FMA + AVX-512; g++ 13.3,
clang++ 18.1 (`machine.txt`). Starting point: the kit's `opt/` (already -73 % vs the reference on an i7-3770).
Files: `STEPS.md` (every step with before/after and parity), `final_timings.txt`, `parity.txt`,
`real_binary.txt`, `baseline.txt` (run.sh), `tools/` (the measuring scripts; they expect to sit in work/tools after run.sh).

## What changed in opt/ and why
1. **Pulse train without libm** (was 2 double sincos per sample). The phase is in cycles, so the quarter turn
   is taken off exactly and sin/cos are Taylor series to x^15 / x^16 in double on |x| <= pi/4 (max abs error
   1.8e-16). Only a plain running sum stays serial; the rest of a frame is one loop without calls or branches,
   which vectorizes. `std::floor` is replaced by an exact +-1.5*2^52 rounding (GCC vectorizes floor only with
   -fno-trapping-math); the gain sqrt(2/harmonics) is sqrt(f) as a geometric series (sqrt may set errno and
   blocks vectorization). Double precision is kept: the pulse matches the reference as well as before
   (158..194 dB). A float version (exact reduction in double, trig in float) was measured and rejected.
2. **Spectrum loop without libm**: `exp_bounded` (float, ln 2 in two parts, Taylor r^7, exponent bits; max rel
   error 1.7 x 2^-24) and `sincos_bounded` (float, pi/2 in three parts, Taylor r^9/r^10; max abs error 8.5e-8,
   libm fallback for any row with |angle| > 8192). Both vectorize.
3. **Noise branch: one exp per envelope** instead of one per filter frame: exp(a + g) = exp(a) exp(g) while no
   bin of the frame is clamped (checked), else the direct path. Neutral with AVX-512, -9 % at baseline ISA.
4. **Linear head in register tiles**: 32 frames at a time with the states transposed, the inner loop along the
   frames; each output is still a dot product in channel order (bit-identical). Weights stay as loaded.
5. **Framing** without the block-wide zero fill and per-sample bounds tests (bit-identical).
6. **Overlap-add**: the sum of squared windows comes from a table added in the same frame order (no per-frame
   accumulation), and the overlap-add buffer is one block long: finished samples are normalized into `out`
   after each block. `shape()` no longer allocates an utterance-long buffer besides the head and `out`.
7. **FFT threads = 1 explicitly**: audio.cpp does not define ENGINE_SPEECHFFT_NO_MULTITHREADING, so the default
   `threads = 0` started the framework's pool for every block; -5 % even on one core.
8. Robustness: under `-ffast-math` (`__FAST_MATH__`) the rounding helpers use std::nearbyint; the shift trick
   alone would be folded away (a fast-math build gave -46.8 dB before, 116.8 dB / 1 LSB after).

Rejected after measuring (STEPS.md): carrying envelopes between blocks, 8 channels or 2 frames per head pass,
head tiles of 16 or 64, block sizes 32/128/256, lower pulse polynomial degree, 4 phase chains per frame, float
pulse trig, -ffast-math.

## Harness results (bench_ac.cpp, 8 real dumps, 3067 acoustic frames; median of 15 interleaved rounds x 15 runs)
us per acoustic frame of `synthesize()` (pulse_train() alone in brackets):

| build | reference | kit opt/ | final opt/ | final / ref | final / kit |
|---|---|---|---|---|---|
| g++ -O3 -march=native, NO_MULTITHREADING (harness default) | 145.52 (17.63) | 35.11 (8.87) | **12.62** (1.79) | 0.087 | 0.36 |
| g++ -O3 -march=native, threaded FFT | 147.05 | 34.75 | 13.29 | 0.088 | 0.38 |
| g++ -O3 (no -march = how audio.cpp builds it), NO_MULTITHREADING | 109.62 (17.00) | 33.69 (9.33) | **26.87** (6.01) | 0.236 | 0.80 |
| g++ -O3, threaded FFT (in-process pairs, 30 iterations) | | 40.56 | 28.63 | | 0.70 |
| clang++ -O3 -march=native | 172.22 | 48.46 | 25.92 | 0.149 | 0.53 |
| clang++ -O3 | 133.85 | 37.72 | 21.77 | 0.165 | 0.58 |

Per case, g++ native (ms; reference / kit / final): u0vF2 10.82 / 2.61 / 1.04, u0vM2 10.02 / 2.40 / 0.99,
u1vF2 47.62 / 10.87 / 4.24, u1vM2 42.99 / 9.91 / 3.63, u2vF2 79.35 / 17.65 / 6.79, u2vM2 71.30 / 17.26 / 6.29,
u3vF2 96.16 / 22.70 / 8.22, u3vM2 86.08 / 19.68 / 7.29. All cases are in `final_timings.txt`.

## Parity (parity.txt)
- Final vs reference, every dump: same length, **min SNR 124.8 dB, max 1 LSB, 916 of 785152 16-bit samples
  differ** (kit opt/: 124.8 dB, 1 LSB, 901). g++ -O3: 127.5 dB / 1 LSB; clang native 124.8 / 1; clang -O3
  127.5 / 1 (also against the same-compiler reference: >= 127.8 dB, 1 LSB).
- Inputs cut to 1 .. 590 frames: worst 122.2 dB, 1 LSB. Pulse train alone vs the reference: 158..194 dB.
- Deterministic (repeat runs bit-identical; identical with and without the threaded FFT). C++17, ASCII, no
  intrinsics, no new dependency; clean with audio.cpp's `-Wall -Wextra -Wpedantic -pedantic-errors`.

## Real audio.cpp binary (real_binary.txt)
audio.cpp 337af0c, CPU only, Release, built unchanged, with the kit's opt/ and with the final opt/. The tiny-voca
unit test passes on all. `audiocpp_cli --task tts --family tiny_voca`, voices M2 and F2, 3 texts (2, 5 and
28-31 s of audio), 10 interleaved rounds:

| case (audio) | unchanged ms (RTF) | kit opt/ | final opt/ |
|---|---|---|---|
| short M2 (2.0 s) | 143.1 (0.070) | 123.2 (0.061) | 119.2 (0.058) |
| short F2 (2.3 s) | 156.9 (0.067) | 130.3 (0.056) | 119.2 (0.051) |
| medium M2 (4.8 s) | 268.1 (0.056) | 201.5 (0.042) | 190.4 (0.040) |
| medium F2 (5.2 s) | 253.6 (0.049) | 224.9 (0.044) | 218.0 (0.042) |
| long M2 (27.9 s) | 1140.0 (0.041) | 895.6 (0.032) | 793.5 (0.028) |
| long F2 (30.5 s) | 1284.7 (0.042) | 985.3 (0.032) | 898.1 (0.029) |
| total, paired ratio to unchanged | 3255 ms | 2601 ms, 0.779 | **2420 ms, 0.735** |

Peak RSS 63-64 MB in all three (the model dominates). WAVs: same length, max 1 LSB, 0.05-0.09 % of samples
differ (SNR on the 16-bit WAVs 101-104 dB, which is what 1-LSB flips on that many samples give). Every binary
was deterministic across rounds.

## Variants (not in opt/, they break a constraint)
- `variants/target_clones`: GCC/Clang function multiversioning on `pulse_train()` and `shape()`. At the real
  build's flags: harness 27.26 -> 13.19 us/frame (ratio 0.48), parity 124.8 dB / 1 LSB; end to end the CLI total
  goes to 0.694 of the unchanged binary (opt/: 0.749 in the same 5 rounds).
- `-ffast-math` on the final source: 116.8 dB / 1 LSB, ratio 0.96 (native) and 1.04 (-O3): no use.

## Risks
- The polynomials and the rounding trick assume IEEE float/double with round-to-nearest (the default
  everywhere audio.cpp runs). `-ffast-math` is handled (S17); a changed FP rounding mode is not.
- Bit patterns still depend on FMA contraction, as before: the -O3 and native builds differ in the last bit on a
  few samples (both well inside parity).
- The speed of the head tile (S12) depends on the vectorizer's plan: tile 16 was 2.25x slower with GCC;
  32 was checked with g++ and clang++, SSE2, AVX2 and AVX-512.
- exp_bounded() casts its rounded exponent to int: a NaN log magnitude (which the reference would pass
  through as NaN) gives garbage instead; only a broken model produces NaN there.

## Left
- At the real build's flags (no -march) the head is 0.80 of the kit's (0.70 with the default threaded FFT):
  most vector gains need wider ISA. Compiling this file (and fft.cpp) with native or multiversioned code,
  as in the variant, is the remaining big step and a build decision for audio.cpp.
- With clang and AVX-512, 57 % of the head's time is the framework FFT's general_c2r (a codegen problem in the
  framework, also slowing the reference and the kit); not touched (framework FFT kept as required).
- The FFTs are now ~40 % of the native time; the number and size of transforms are fixed by the method.
