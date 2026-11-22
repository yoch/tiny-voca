# Summary: a lean patch for the tiny-voca waveform head

Machine: Intel Xeon @ 2.10 GHz, 4 vCPU KVM (AVX2/FMA/AVX-512 present but NOT used by the real build), g++ 13.3,
clang++ 18.1, glibc 2.39 (`machine.txt`). Target: audio.cpp 337af0c, `src/community_models/tiny_voca/source_filter.cpp`.
All numbers are at audio.cpp's real flags (`-O3 -DNDEBUG -std=c++17 -fopenmp`, no -march, threaded framework FFT)
unless marked; method and the full table in `ABLATION.md`.

## Recommendation: LEAN (`lean/lean/`, `lean/lean/lean.patch`)
**148 changed lines (132 code + 16 comment/blank), no new helper function (one removed), no special case,
no polynomial, every libm call kept; 0.294 of the reference's time, 90.7 % of the final's saving.**

It is four independent changes plus one setting, each readable alone:
1. **The filter response by the framework FFT** (48 lines): the reference builds each filter frame's cepstrum and
   then evaluates its transform by a direct double DFT (96 x 257 multiply-adds twice, from two tables) before
   exp/cos/sin. LEAN keeps the cepstrum code as it is, writes the cepstra of a block into frames of `fft` samples
   and transforms them in one batched call of the same real FFT plan already used for the signal; the bin loop
   becomes exp(clamp(re)), cos(im), sin(im) in float. The tables and `response_tables()` go. 0.457 alone.
2. **FFT threads = 1** (8 lines): the framework's default (0) hands every block of short transforms to its pool,
   which costs more than it saves (0.908 on top of 1).
3. **Pulse train with one exp per frame** (36 lines, the kit's): the log-linear pitch is a geometric series inside a
   frame; the phase is in cycles; cos((K + 1) x) from the two angles already used, so 2 sincos per sample instead
   of exp + fmod + 3 trig (0.817 on top of 1 + 2).
4. **Linear head with transposed weights** (32 lines, the kit's): the loop runs along the outputs and vectorizes;
   each sum is still added in channel order (bit-identical) (0.888).
5. **Overlap-add one block long** (23 lines): the two utterance-long buffers become block * hop + fft, finished
   samples go to `out` after each block; same additions in the same order, bit-identical (0.955; it also makes the
   memory bounded by blocks, as the comment of `shape()` already claimed).

Why LEAN and not MID or FAST: the head is about a quarter of the CLI's wall time, so the 1.29x that separates LEAN
from the final in the head is 1 % of the end-to-end time (real binary below: 2492 vs 2467 ms), while MID adds two float
polynomials, a fallback path and a loop split that exists only for GCC's vectorizer, and FAST adds three more
helpers and a `__FAST_MATH__` case. LEAN's correctness argument is short: same cepstra, same clamp, same libm
functions (float instead of double), the FFT computes the same sum as the removed DFT. Its parity is the best of all
versions (129.2 dB vs 127.8 dB for kit and final).

## Candidates
| | lines changed (code + comment) | new helpers | special cases | us/frame (real flags) | vs reference | vs final | -march=x86-64-v3: vs reference / vs final | parity vs reference (min SNR, max, #differ of 785152) |
|---|---|---|---|---|---|---|---|---|
| reference (audio.cpp 337af0c) | 0 | | | 125 .. 140 | 1 | | 1 | |
| kit (50222d1) | 156 (138 + 18) | 0 | 0 | 41.9 | 0.307 | | | 127.8 dB, 1, 688 |
| final (opt/) | 426 (344 + 82) | 7 + 2 lambdas | `__FAST_MATH__` x2, sincos range fallback, noise clamp test | 28.6 | 0.222 | 1 | 0.159 / 1 | 127.8 dB, 1, 693 |
| **LEAN** | **148 (132 + 16)** | **0 (1 removed)** | **none** | **40.6** | **0.294** | **1.285** | 0.241 / 1.468 | **129.2 dB, 1, 617** |
| MID = LEAN + vectorized exp/sincos in float | 201 (171 + 30) | 2 | sincos range fallback | 31.4 | 0.252 | 1.061 | 0.179 / 1.103 | 129.2 dB, 1, 622 |
| FAST = MID + final's pulse train | 270 (219 + 51) | 5 | range fallback, `__FAST_MATH__` | 26.7 | 0.214 | 0.927 | 0.151 / 0.944 | 129.2 dB, 1, 622 |

us/frame: B of the 40-iteration pairs against the reference (the reference read 125 .. 140 us/frame in those runs).
Ratios are paired medians (quartiles in ABLATION.md). clang++ 18 at the real flags: LEAN 0.286, MID 0.224, FAST 0.206, final 0.165 of the clang-built reference (clang vectorizes the final's tiled loops better than GCC; LEAN / final 1.675).

Optional on LEAN: sharing the noise phase between the 4 filter frames of a held envelope (the gain only moves the
real part of the log response) is +5 lines for 0.970 (`ABLATION.md`, LEAN + B'); left out to keep LEAN free of any
reuse logic.

## Checks (all three candidates)
- Parity vs the reference built with the same flags, every dump: same length, SNR >= 129.2 dB, max 1 LSB
  (vs the -march=native reference of run.sh: 128.8 dB, 1 LSB). clang++: the same (129.2 dB, 1 LSB) against both
  the clang and the g++ reference.
- Short inputs (1 .. 590 frames, `pair MODE=edge`): worst 125.0 dB, max 1 LSB for all three (final: 123.9 dB, 1 LSB).
- Pulse train alone vs the reference: min SNR 158.0 dB, max abs difference 9.5e-7 for all three (the same as kit and final).
- Deterministic: two runs bit-identical. `-ffast-math` build (not required, robustness only): LEAN 128.6 dB,
  MID / FAST 128.4 dB, 1 LSB (final: 117.9 dB).
- Clean with `-O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -pedantic-errors -fopenmp` under g++ 13 and clang++ 18
  (no output). ASCII. C++17, no intrinsics, no target attributes, no new dependency, framework FFT kept, memory
  bounded by blocks (the head and the pulse train stay utterance-long, as in the reference and the final).
- `git apply --check` of each patch on a clean audio.cpp 337af0c: OK.

## Real audio.cpp binary
`real_binary.txt`. audio.cpp 337af0c built as audio.cpp builds it (CPU, Release; this file gets
`-O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -pedantic-errors -fopenmp`, checked in flags.make; no warnings),
unchanged and with LEAN, MID and the final copied in. `tiny_voca_unit_test` passes for all four. `audiocpp_cli --task tts
--family tiny_voca`, voices M2 / F2, three texts, 10 interleaved rounds:

| case (audio) | unchanged ms (RTF) | LEAN | MID | final |
|---|---|---|---|---|
| short M2 (2.0 s) | 145.2 (0.071) | 128.5 (0.063) | 119.5 (0.059) | 124.2 (0.061) |
| short F2 (2.3 s) | 153.3 (0.065) | 127.6 (0.054) | 127.7 (0.054) | 127.8 (0.055) |
| medium M2 (4.8 s) | 281.8 (0.059) | 215.8 (0.045) | 213.9 (0.045) | 194.5 (0.041) |
| medium F2 (5.2 s) | 276.4 (0.054) | 213.3 (0.041) | 214.4 (0.042) | 222.9 (0.043) |
| long M2 (27.9 s) | 1247.7 (0.045) | 897.0 (0.032) | 833.6 (0.030) | 848.0 (0.030) |
| long F2 (30.5 s) | 1309.3 (0.043) | 918.9 (0.030) | 948.9 (0.031) | 918.9 (0.030) |
| total (median), paired ratio to unchanged | 3414 ms | **2492 ms, 0.738** | 2464 ms, 0.714 | 2467 ms, 0.723 |

End to end the three are within 1 % of each other (LEAN total / final total = 1.010; the paired ratios spread
0.65 .. 0.86 per round on this VM, so the 0.738 / 0.714 / 0.723 medians do not separate them). Peak RSS 62.8 - 64.2 MB
for all (the model dominates). WAVs: same length, max 1 LSB, LEAN 33 - 559 differing samples per file (fewer than the
final's 50 - 640), SNR on the 16-bit WAVs 102.4 - 104.9 dB. Every binary was deterministic across rounds.

## Deviations
- The audio.cpp build ran on the 3 other cores during the last timing batches. Pairs are paired and pinned, so A and
  B saw the same load, but absolute us/frame drift (the reference read 115 .. 162 us/frame across runs); only ratios
  are compared.
- Two early pairs overlapped with another batch on the pinned core by mistake (b1 kit -> final, b3 A1ECDF ->
  final; the raw files keep them); every number the candidates are judged by was re-measured in the final 40-iteration
  batch (`raw/f_real.txt`).
- clang++ pair binaries were compiled with `-fopenmp` but linked without it (no libomp on this machine; nothing in
  these files uses OpenMP). The clang parity binaries were built with `-O3` only, for the same reason.
- The final's own pieces beyond FAST (register-tiled head, framing rewrite, window-square table, noise exp per
  envelope, phase per envelope) were measured as one group (final / FAST = 1.079 at the real flags), not one by one:
  together they are slower than FAST with GCC at the real flags, so splitting them could not change the choice.
  F, G, P, K, Fx were measured on top of the A1 combination only (they modify code A1 introduces).
- FAST was not built into the real binary (harness only); LEAN, MID and the final were.
- "Memory bounded by blocks": the overlap-add is block-sized in all three candidates (as in the final); the linear head
  output and the pulse train stay utterance-long, as in the reference and in the final.
- `run.sh` rewrote `results/baseline.txt`; it was restored from git (that run's output is in `raw/baseline_runsh.txt`).
- Parity reference: the reference built with the same flags (`-O3 -fopenmp`); against run.sh's `-march=native`
  reference the numbers are also given (128.8 dB, 1 LSB).
