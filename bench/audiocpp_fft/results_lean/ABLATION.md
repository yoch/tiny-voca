# Ablation: tiny-voca waveform head at audio.cpp's real build flags

Machine: `machine.txt` (Intel Xeon @ 2.10 GHz, 4 vCPU KVM, g++ 13.3, clang++ 18.1, glibc 2.39).

## How it was measured
- **Real flags** (the decisive column): `g++ -O3 -fopenmp -std=c++17 -DNDEBUG`, no `-march` (baseline x86-64, SSE2),
  `ENGINE_SPEECHFFT_NO_MULTITHREADING` NOT defined, so the framework FFT is threaded by default exactly as in
  audio.cpp. fft.cpp is compiled with the same flags. Secondary column: the same plus `-march=x86-64-v3` (AVX2 + FMA).
- **Timing**: `tools/pr.sh A B 30|40` (wraps `results/tools/pair.cpp`): A and B are linked into one binary, every
  iteration runs the 8 real dumps (3067 acoustic frames) with A then B (order swapped each iteration), pinned to
  core 2. Reported: median of the per-iteration ratio B/A (quartiles), 30 iterations (40 for the final candidates).
  Absolute us/frame drift by +-20 % between runs on this VM (the reference read 115 .. 162 us/frame); the ratios
  are what decide. The audio.cpp build was running on the other three cores during the last batches (common to
  A and B).
- **Lines**: `tools/lines.sh` = `diff -u` of source_filter.h + source_filter.cpp against audio.cpp 337af0c, lines
  starting with `-` or `+` (file headers excluded), split into code and comment-only/blank lines.
- **Parity**: `tools/par.sh` (bench_ac.cpp + compare.py) against the reference built with the same flags
  (`r_ref`), and against the `-march=native` reference of run.sh (`out_ref`).

All variant sources are in `work/abl/<name>/` (not committed; rebuilt from the descriptions below and the tools in
`results_lean/tools`). The candidates are committed in `lean/`.

## The ideas
| id | idea | source | lines (code + comment) |
|---|---|---|---|
| A1 | **new**: the response of each filter frame by the framework's real FFT. The cepstra of the block are written (interpolated, liftered, gain on c0, exactly as the reference builds them) into frames of `fft` samples and transformed in one batched call; then per bin exp(clamp(re)) and cos/sin(im) in float, libm. The O(coefficients x bins) double DFT and its tables go | simpler form of kit 1 | 44 + 4 = 48 |
| A | kit's form of the same: one FFT per acoustic-frame envelope, log responses interpolated, gain added after | kit 1 | 75 + 13 = 88 |
| B | phase once per envelope (on A) | kit 2 | 0 (on A) |
| E | FFT `threads = 1` on the framework calls (the default 0 starts the pool for every block) | final 7 | 5 + 3 = 8 |
| C | linear head with transposed weights, loop along the outputs (bit-identical sums) | kit 3 | 29 + 3 = 32 |
| D | pulse train: one exp per frame (geometric series), phase in cycles, 2 sincos instead of 3 trig + exp + fmod per sample | kit 4 | 34 + 2 = 36 |
| D1 | **new**: pulse train, only fmod replaced by a phase in cycles | new | 6 + 1 = 7 |
| K | complex product written out instead of `std::complex` `*=` | kit | 2 + 1 = 3 |
| M | **new**: overlap-add buffers one block long (block * hop + fft) instead of the utterance; finished samples are normalized into `out` after each block. Same additions in the same order: bit-identical | simpler form of final 6 + 11 | 20 + 3 = 23 |
| Fx | **new**: exp by polynomial only (libm cos/sin stay, so the loop stays scalar) | new | 10 + 3 = 13 |
| F | **new form** of final 2: `exp_small` / `sincos_small` in float, rounding by int32 conversion of an offset value (no `__FAST_MATH__` case), clamp in its own loop (next to the polynomial it blocks GCC's vectorizer), libm fallback for a row with abs(angle) > 8192. Both loops vectorize at SSE2 | final 2 | 39 + 14 = 53 |
| P | **new**: the pulse sin/cos by a scalar double Taylor series (no vector passes) | new | 13 + 5 = 18 |
| G | final's pulse train (serial phase pass + vectorized Taylor pass in double, `round_small` with its `__FAST_MATH__` case), double helpers only | final 1 | 48 + 21 = 69 |
| rest | final's register-tiled head, framing without zero fill, window-square table, noise exp per envelope, phase per envelope | final 3 - 6 | (final has 344 + 82 = 426 in all) |

## Results at the real flags (g++, ratio = paired median B/A)
Alone on the reference:

| change | ratio vs reference | note |
|---|---|---|
| E alone | 0.972 (0.960 .. 0.980) | |
| C alone | 0.964 (0.947 .. 0.990) | |
| D alone | 0.935 (0.923 .. 0.956) | |
| D1 alone | 1.005 (0.984 .. 1.025) | fmod is not the cost: rejected |
| A alone | 0.460 (0.447 .. 0.468) | |
| A + B | 0.443 (0.429 .. 0.457) | |
| **A1 alone** | **0.457 (0.434 .. 0.477)** | as fast as A with half the lines |
| M alone | 0.981 (0.964 .. 0.996) | bit-identical |

In combination (each step against the previous one; "pts" = percentage points of the reference time saved,
from the ratios vs the reference):

| step | ratio vs previous | ratio vs reference | lines | pts per 10 lines |
|---|---|---|---|---|
| A1 | 0.457 | 0.457 | 48 | 11.3 |
| + E | 0.908 | 0.415 | 56 | 5.3 |
| + K (written-out product) | 0.94 (A1E), 1.00 (on A1ECD) | 0.390 / 0.300 | +3 | not kept: no gain in the full combination |
| + D | 0.817 | 0.339 | 94 | 2.1 |
| + C | 0.888 | 0.301 (0.306 in a second run) | 124 | 1.3 |
| + M = **LEAN** | 0.955 (0.936 .. 0.978) | **0.294** (0.285 .. 0.303) | **148** | 0.6 (and the memory bound) |
| + Fx (exp polynomial only) | 1.079 | | +13 | slower: rejected |
| + F = **MID** | 0.825 (on A1ECD) | **0.252** (0.246 .. 0.259) | **201** | 0.8 |
| + P (scalar pulse Taylor) | 0.968 (on A1ECDF) | | +18 | 0.5 |
| + G = **FAST** | 0.879 (on A1ECDF) | **0.214** (0.207 .. 0.219) | **270** | 0.55 |
| LEAN + B' (noise phase shared by the 4 filter frames of a held envelope) | 0.970 (0.942 .. 0.991) on LEAN | | +5 | 0.9; optional, not in LEAN (see SUMMARY) |
| final (opt/) | final / FAST 1.079 | 0.222 (0.214 .. 0.232); 0.219 in an earlier run | 426 | the 157 extra lines lose 8 % |
| kit (50222d1) | | 0.307 .. 0.318 | 156 | |
| kit + E | 0.916 (on kit) | 0.285 | 164 | |
| kit + E + G | 0.898 (on kit + E) | | 245 | |
| A1ECDF + G (final's helpers, float ones too) | | 0.233 | 262 | same speed as final (final/that 1.014) |

Candidates against the final (real flags, 40 iterations): LEAN / final 1.285 (1.251 .. 1.332), MID / final 1.061
(1.048 .. 1.087), FAST / final 0.927 (0.914 .. 0.936).

Share of the final's saved time (1 - ratio vs reference, over the final's 1 - 0.222 = 0.778): LEAN 90.7 %,
MID 96.1 %, FAST 101 %.

## Secondary column: -march=x86-64-v3 (AVX2 + FMA), g++
30 iterations, same pairs (ratio B/A):

| | vs reference | vs final |
|---|---|---|
| kit | 0.244 (0.239 .. 0.255) | |
| final | 0.159 (0.153 .. 0.163) | 1 |
| LEAN | 0.241 (0.231 .. 0.247) | 1.468 (1.449 .. 1.480) |
| MID | 0.179 (0.172 .. 0.181) | 1.103 (1.020 .. 1.128) |
| FAST | 0.151 (0.147 .. 0.156) | 0.944 (0.911 .. 0.974) |

With AVX2 the vectorized pieces (F, G) are worth more, so LEAN keeps a smaller share of the final's saving
(90.3 % of it: 0.759 / 0.841).

## clang++ 18, real flags
clang++ 18 (`-O3 -fopenmp`, no -march; linked without -fopenmp, no libomp is installed), 30 iterations:

| | vs reference | vs final |
|---|---|---|
| final | 0.165 (0.162 .. 0.166) | 1 |
| LEAN | 0.286 (0.278 .. 0.294) | 1.675 (1.654 .. 1.722) |
| MID | 0.224 (0.219 .. 0.228) | 1.397 (1.345 .. 1.421) |
| FAST | 0.206 (0.199 .. 0.208) | |

clang vectorizes the final's register-tiled head and its other rewritten loops better than GCC does, so with clang
the final's extra pieces do pay (LEAN keeps 85.5 % of the final's saving: 0.714 / 0.835). audio.cpp's default
toolchain on Linux is GCC; the clang numbers are a secondary data point.

## Why the final's other pieces do not pay at the real flags
At SSE2 the vector width is 4 floats / 2 doubles, so the work that remains after A1 + E + C + D is dominated by the
libm calls per bin (expf + sincosf, 1030 per acoustic frame) and per pulse sample (2 sincos): F and G remove those,
and that is the whole gap to the final. The register-tiled head, the window-square table, the framing rewrite, the
noise exp per envelope and the phase per envelope (together ~157 changed lines in the final beyond FAST) are slower
in combination at these flags (FAST is 7 % faster than the final). Two details mattered for vectorization with
GCC at -O3 without fast-math: the clamp `std::min(std::max(x, -14), 6)` in the same loop as the polynomial is jump-
threaded into branches with trapping multiplies (no vectorization), so it gets its own loop; and quadrant selects
must choose between already computed values (a select of `x * sign` is split into branches).
