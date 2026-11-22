# Step log: tiny-voca waveform head (opt/src/source_filter.cpp)

Machine: see `machine.txt` (Xeon @ 2.10 GHz, 4 vCPU KVM/Firecracker, AVX-512, gcc 13.3, clang 18.1).

How every step was measured:
- **Parity** (`compare.py` against the reference outputs, the 8 real dumps): min SNR over the dumps, max 16-bit
  difference, number of 16-bit samples that differ. The kit's opt/ gives 124.8 dB, 1 LSB, 901 samples.
- **Speed**: the VM's speed drifts by +-20 % over seconds, so medians of separate processes were not enough to see
  5 % steps (quartiles of the ratio 0.82 .. 1.02). From S2 on, every step was timed with `work/tools/pair.cpp`:
  the previous step (A) and the candidate (B) are linked into one binary (namespace renaming), every iteration
  runs all 8 cases with A then B (order swapped each iteration), pinned to one core; reported: median us per
  acoustic frame of A and B and the median (quartiles) of the per-iteration ratio B/A over 30 iterations.
  The ratio is what decides; absolute us/frame drift from one run to the next.
- Default build of a step: `g++ -O3 -march=native -std=c++17 -DNDEBUG -DENGINE_SPEECHFFT_NO_MULTITHREADING`
  (the kit's harness flags). Steps that matter for the real audio.cpp build (which compiles this file with
  `-O3 -DNDEBUG` and no -march, see SUMMARY.md) were also timed with `-O3` only (SSE2).
- Error of every polynomial: separate tests against libm in double / long double (`work/t/sc.cpp`, `work/t/fl.cpp`);
  pulse train against the reference pulse train (`pair MODE=pulse`); short inputs (1 to 590 frames, cut from the
  longest dump) against the reference (`pair MODE=edge`).

| step | idea | change | time before -> after (us/frame) | ratio (quartiles) | parity (min SNR, max, #differ) | kept |
|---|---|---|---|---|---|---|
| S0 | baseline | reference / kit opt/ (cmp.py 5 x 15) | ref 144.34, opt 34.74 (pulse 17.63 / 8.70) | | 124.8 dB, 1 LSB, 901 | |
| S1 | pulse train: no libm sincos | phase pass (serial) + vector pass; `sincos_cycles`: quarter turn taken off exactly in cycles, Taylor x^15 / x^16 in double (max abs error 1.8e-16 vs long double, 2e7 samples); `round_small` (+-1.5*2^52) instead of `std::floor`; sqrt kept in the serial pass | 31.11 -> 27.24 (pulse 7.84 -> 2.82), cmp.py | | 124.8, 1, 901 (identical) | yes |
| | | tries that did not vectorize: integer quadrant (mixed widths), `std::floor` (GCC needs -fno-trapping-math to vectorize it), sqrt in the vector loop (may set errno) | | | | |
| S2 | no per-bin `std::exp` | `exp_bounded`: n = round(x log2 e), ln 2 in two parts, Taylor r^7, 2^n written into the exponent (max rel error 1.02e-7 = 1.7 x 2^-24 on [-40, 40]; std::exp(float) 1.0 x 2^-24) | 28.52 -> 25.09 | 0.868 (0.852..0.883) | 124.8, 1, 897 | yes |
| S3 | no per-bin `std::cos/sin` | `sincos_bounded`: q = round(x 2/pi), pi/2 in three parts (Cephes), Taylor r^9 / r^10 (max abs error 8.5e-8 for abs(x) <= 8192); rows with any abs(angle) > 8192 fall back to libm. First version used `a || b` for the range flag: control flow, not vectorized, ratio 1.030; with a count it vectorizes | 24.60 -> 20.15 | 0.812 (0.797..0.830) | 124.8, 1, 902 | yes |
| S4 | head: fewer loads/stores | 4 channels per pass along the outputs, same summation order | 16.03 -> 14.59 | 0.911 (0.899..0.928) | bit-identical to S3 | yes (replaced by S12) |
| S5 | framing without zero fill and bounds tests | zero only the parts of a frame outside the source | 16.41 -> 15.33 | 0.925 (0.913..0.934) | bit-identical | yes |
| S6 | precomputed window products | sum of squared windows from a table of `fft` values, added in frame order; the per-frame `envelope += w*w` and its utterance-long array are gone | 15.22 -> 14.06 | 0.933 (0.923..0.940) | bit-identical with -ffp-contract=off; with contraction 911 samples (the reference's `envelope += w*w` becomes an FMA), SNR 124.8, 1 LSB | yes |
| S7 | whole SIMD batches of envelope FFTs | carry the shared harmonic envelope to the next block, no "following" envelope for the held noise (32/16 instead of 33/17 per block) | 15.06 -> 14.97 | 0.997 (0.985..1.005) | bit-identical | no |
| S8 | pulse: shorter serial chain | serial pass = plain running sum of phase steps (no floor, reduced in the vector pass); gain = sqrt(2/(limit rate)) sqrt(f) with sqrt(f) as a geometric series | 15.04 -> 14.02 | 0.935 (0.927..0.952) | 124.8, 1, 911; pulse vs reference pulse 158..194 dB, max abs 9.5e-7 (same as kit) | yes |
| S9 | head, 8 channels per pass | | 14.26 -> 14.37 | 1.000 (0.986..1.015) | bit-identical | no |
| S9b | head, 2 frames x 4 channels | lambda, possible aliasing: not vectorized | 14.37 -> 17.74 | 1.242 | bit-identical | no |
| S10 | FFT threads = 1 passed explicitly | `kFftThreads = 1` on the 3 calls; measured WITHOUT ENGINE_SPEECHFFT_NO_MULTITHREADING (the real build) | pinned 15.44 -> 14.57; unpinned (4 cores) 15.31 -> 14.56 | 0.946 (0.930..0.958); 0.950 (0.938..0.964) | identical | yes |
| | batching (kBlockFrames) | 32 / 128 / 256 vs 64 | | 1.028 / 1.051 / 1.129 | | 64 kept |
| S11 | memory bounded by blocks | overlap-add buffer of one block (block*hop + fft) instead of the utterance; finished samples normalized into `out` after each block | 14.11 -> 13.77 | 0.981 (0.972..1.002) | bit-identical | yes |
| S12 | head: register tiles | 32 frames per tile, states transposed (channels x frames), inner loop along the frames, sums in registers; still one dot product per output in channel order; weights kept as loaded | 13.76 -> 13.15 | 0.954 (0.942..0.966); gcc -O3 0.960, gcc -march=x86-64-v3 0.975, clang native 0.995, clang -O3 0.944; tile 64: 1.009, tile 16: 2.250 | bit-identical | yes |
| S13 | pulse: lower degree | Taylor x^13 / x^14 (error 2e-14) | 13.38 -> 13.17 | 1.002 (0.988..1.019) | | no |
| S14 | pulse: 4 independent chains per frame | | 12.48 -> 12.34 | 1.003 (0.993..1.020) | 124.8, 1, 912 | no |
| S15 | pulse: float trig after exact reduction in double | `sincos_cycles_float`, closed form in float | native 10.22 -> 12.62; -O3 31.42 -> 30.15 | native 1.231 (mixed widths vectorize badly); -O3 0.988 | 124.1, 1, 962; pulse 142 dB, max abs 2.9e-6 | no |
| S16 | noise: one exp per envelope | held envelope: exp(a + gain) = exp(a) exp(gain) when no bin of the frame is clamped and abs(a), abs(gain) <= 40 (counting pass), else the direct path | native 13.99 -> 13.90; -O3 30.58 -> 27.85 | native 1.005 (0.985..1.030); -O3 0.906 (0.893..0.923) | 124.8, 1, 916 | yes (helps the real build) |
| S17 | robustness under -ffast-math | rounding helpers use std::nearbyint under `__FAST_MATH__` (else the shift folds away: -ffast-math build gave SNR -46.8 dB; after: 116.8 dB, 1 LSB) | | | bit-identical to S16 | yes |
| S18 | comments, header | lifter comment back above lifter(), constants grouped, header comment on the weight layout | | | bit-identical | yes |

Variants (constraint-breaking, under `variants/`, never in opt/): see `final_timings.txt` and SUMMARY.md.
- `-ffast-math` on the final source: parity 116.8 dB, 1 LSB, 2027 samples; ratio vs final 0.961 (native),
  1.039 (-O3): no use.
- `variants/target_clones/source_filter.cpp`: GCC/Clang function multiversioning (x86-64-v4, v3, default) on
  `pulse_train()` and `shape()`; parity 124.8 dB, 1 LSB, 941 samples.

Checks of the final candidate (`opt/`):
- parity on every dump: see `parity.txt`; same compiler pairs (g++/clang++, native/-O3) all >= 124.8 dB, 1 LSB.
- short inputs 1 .. 590 frames vs reference: worst 122.2 dB (native) / 123.9 dB (-O3), max 1 LSB.
- deterministic: three runs bit-identical; identical with and without ENGINE_SPEECHFFT_NO_MULTITHREADING.
- builds without warnings with audio.cpp's flags (`-O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic
  -pedantic-errors -fopenmp`), g++ 13 and clang++ 18.
