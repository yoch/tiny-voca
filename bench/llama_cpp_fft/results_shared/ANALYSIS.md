# Shared mtmd FFT: analysis

Can the shared FFT of `tools/mtmd/mtmd-audio.cpp` (`fft_impl` / `dft_impl` behind `fft()`, `ifft()`,
`mtmd_audio_fft()`, `mtmd_audio_ifft()`) be made fast enough, with a small, numerically safe patch, that tiny-voca
(unchanged model code) and the other callers get the speed-up? Patches: `../shared/shared-fft-exact.patch` and
`../shared/shared-fft-packed.patch`; patched files: `../shared/src/mtmd-audio.{exact,packed}.cpp`.

## Setup
- llama.cpp master `8d60873dfb8619f2b5ed31adfbab07cccb40db5c` (2026-11-22 13:40 +0200). `mtmd-audio.cpp`,
  `mtmd-audio.h`, `models/tiny-voca.cpp` and `mtmd.cpp` are byte-identical to `e300919cc` (previous session), so
  the LEAN / MID patches apply unchanged (`git apply --check` clean).
- Machine: 4-vCPU KVM cloud VM, Intel Xeon @ 2.10GHz (Emerald Rapids class, AVX-512/AVX2/FMA available), 15 GiB,
  Ubuntu 24.04, g++ 13.3.0, clang++ 18.1.3, cmake 3.28.3. Shared VM: noisy neighbours are visible (a master vs
  master control pair gives a median ratio of 1.00 to 1.01, quartiles 0.93-1.09), so every number is an interleaved
  A/B and quartiles are given.
- Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release` with default options. `mtmd-audio.cpp` is its own TU
  (`SKIP_UNITY_BUILD_INCLUSION`), compiled as (`raw/compile_line_mtmd_audio.txt`, `raw/flags.make.mtmd.txt`):

      c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_BUILD -DLLAMA_SHARED -DMTMD_VIDEO
          -D_USE_MATH_DEFINES -Dmtmd_EXPORTS -I... -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn
          -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-cast-qual
          -Wno-maybe-uninitialized

  No `-march` (x86-64 baseline: SSE2, no FMA, so no contraction), g++ 13 default `-std=gnu++17`. All main numbers
  are at these flags; `-march=native` is a secondary column of the standalone harness only.
- `timing-instrumentation.patch` (TV_TIMING, `mtmd.cpp` only) is applied to every llama-tts build, as before.
- Builds (`raw/builds.txt`, all 0 warnings): `master`; `exact` / `packed` (patched `mtmd-audio.cpp`, master
  `tiny-voca.cpp`); `lean` / `mid` (previous candidates, master `mtmd-audio.cpp`); `exact_pc` / `packed_pc`
  (patched `mtmd-audio.cpp` + only MID's pulse-train hunk, `lean/src/Pc.cpp`); `master2` (master rebuilt, control).

## Callers of the FFT and the sizes they use
`fft()` is always forward with real input; `ifft()` always inverse with complex input (the other two template
combinations are unused). N = 2^p * odd.

| caller | function | N (= 2^p x odd) | direction | per |
|---|---|---|---|---|
| tiny-voca host synth (`models/tiny-voca.cpp`, `tiny_voca_filter`) | `mtmd_audio_fft` x2, `mtmd_audio_ifft` x1 | 512 (harmonic), 256 (noise) | fwd real, inv complex | STFT frame (hop 128 / 64) |
| voxtral realtime streaming (`mtmd.cpp`, `mtmd_audio_stream_compute_mel`) | `mtmd_audio_log_mel_frame` -> `fft` | 400 = 16 x 25 | fwd real | mel frame |
| chatterbox speaker ref: s3 tokenizer log-mel (x2), voice-encoder mel | `mtmd_audio_mel_energies` -> `fft` | 400 = 16 x 25 | fwd real | frame |
| chatterbox speaker ref: matcha (s3gen prompt) log-mel | `mtmd_audio_mel_energies` -> `fft` | 1920 = 128 x 15 | fwd real | frame |
| chatterbox speaker ref: kaldi fbank (CAMPPlus) | `mtmd_audio_mel_energies` -> `fft` | 512 | fwd real | frame |
| zonos3 iSTFT head (`mtmd_audio_zonos3_synth::push`) | `ifft` | 736 = 32 x 23, 2208 = 32 x 69 | inv complex | column, both branches |
| `mtmd_audio_streaming_istft::process_frame`: LFM3A_GEN | `ifft` | 1280 = 256 x 5 (hop 320) | inv complex | frame |
| `mtmd_audio_streaming_istft`: Soprano (`models/soprano.cpp`) | `ifft` | 2048 (hop 512; head tensor 2050 rows, read from the GGUF) | inv complex | frame |

Not callers any more: since PR 31808 (merged 2026-11-07) the log-mel of the preprocessors whisper, qwen3a,
dots3note, higgs_audio, mimo_audio (960), qwen3tts_spk (1024), conformer, d1omni, lfm3a (400), granite_speech
(512), gemma4a (512), parakeet / nemotron_transcribe (512), gemma4ua, zonos3_spk (512) and others goes through
`mtmd_audio_log_mel` (cgraph DFT matmul, `mtmd-subcontext.cpp`). They still call `fill_sin_cos_table` (the DFT
matrix reads the table) but not `fft()`. Kokoro's iSTFT has its own direct DFT and the kokoro source STFT
(n_fft 16) is a direct DFT too; neither uses the shared FFT. `tests/test-mtmd-impl.cpp` did not test the FFT
(its log-mel tests go through the cgraph path).

Contracts (mtmd-audio.h): "in needs 2 * N floats and out 8 * N floats (scratch)" for `mtmd_audio_fft`, "in and out
need 8 * N floats" for `mtmd_audio_ifft`; every internal caller allocates 8 N for `out`. The sin/cos table of
`fill_sin_cos_table(T)` is read at index `k * (T / len)`, so T must be a multiple of N (every caller uses T = N).

## Profile of the master FFT
Per level of the recursion it copies the even and odd halves into the scratch (`in + N` or `in + 2N`), recurses
twice, and combines with twiddles read at a stride of T / N from the table, with a 0.5 scale per level for the
inverse. The odd part is a direct DFT whose inner loop computes `(k * n * step) % T` (an integer division per
multiply-add) and is not vectorizable. For 2208 = 32 x 69 that is 32 x 69^2 = 152k scalar complex MACs, each
with a modulo; for 400, 16 x 25^2. Callgrind of the first iterative rewrite (v4 below, 512 forward): computing the
bit-reversed index on the fly was 33 % of all instructions.

## Steps (standalone harness, each variant in its own TU at the real flags, `shared/harness/variants/*.inc`)
Ratios are new / master, in-process ABBA, pinned core; parity = bit-identical outputs on 20 random + 8 real-audio
inputs per size (`shared/harness/micro.cpp`).

| step | idea | 512 fwd | 512 inv | 400 fwd | 2208 inv | 1280 inv | bit-exact |
|---|---|---|---|---|---|---|---|
| v1 | iterative: DFT of each odd-size subsequence read in place (stride 2^p), then in-place radix-2 stages; incremental index instead of `%` in the DFT | 0.90 | 0.87 | 0.66 | 0.71 | 0.72 | yes |
| v2 | + contiguous twiddles of each stage gathered into the scratch | 0.80-0.84 | 0.85 | 0.64 | 0.71 | 0.72 | yes |
| v3 | + first stage (twiddle 1) done while gathering the input in bit-reversed order (n_odd == 1) | 0.46 | 0.52 | 0.64 | 0.71 | 0.71 | yes |
| v4 | + odd-size DFT of all 2^p subsequences at once (vectorizable across subsequences, same per-output summation order) | 0.48 | 0.53 | 0.31 | 0.24 | 0.75 | yes |
| v6 = **exact** | + incremental bit reversal (was 33 % of instructions) | **0.32** | **0.36** | **0.26** | **0.25** | **0.46** | **yes** |
| v7 | + real input packed as N/2 complex values + split (with 3 `%` per bin) | 0.86-0.97 of v6 | = | 0.87-0.90 of v6 | = | = | no (fwd) |
| v8 = **packed** | v7 without the `%` (Z[N/2] = Z[0] stored once) | **0.66-0.70 of v6** (0.22) | = | 0.79-0.81 of v6 | = | = | no (fwd) |
| v9 | + k-outer loop order for stages with fewer than 8 butterflies per block | 1.00 of v8 | 1.00 | 0.99 | 0.95-1.0 | 1.00 | (as v8) |

Single 31-pair runs during development (some while other jobs ran), so the step ratios are indicative; the final
candidates are measured below on the real libraries. v9 brought nothing and was dropped. Further ideas not tried, because they would change the cache layout or add
special cases: per-size precomputed twiddle / bit-reversal tables in `mtmd_audio_cache` (what LEAN's
`tiny_voca_rfft` does), fused radix-4 first stages (exact only with the table's cos(pi/2) = -4.4e-8, so it would
break bit-exactness), mixed radix for the odd part (3 x 23, 3 x 5), and a real-output inverse (needs a new
function: callers use only the real part of `ifft`, but its contract returns N complex values).

## The two candidates
**exact** (`shared-fft-exact.patch`): `dft_impl` + recursive `fft_impl` are replaced by one iterative
`fft_impl`: (1) for n_odd > 1, the DFT of size n_odd of the 2^p subsequences `in[r + 2^p n]`, all r at once, the
result of subsequence r written to block bitrev(r); for n_odd == 1 the first radix-2 stage is done while gathering;
(2) in-place radix-2 stages with the twiddles of the stage copied contiguously into the scratch. Each output value
goes through exactly the same float operations, in the same order, as in master (same table entries, same
expressions, the inverse still scales by 1/n_odd in the DFT and by 0.5 per stage), so at the real flags the output
is bit-identical for every N (only signed zeros could differ; they compare equal).
`fft()` / `ifft()` / `mtmd_audio_fft()` / `mtmd_audio_ifft()` are unchanged.

**packed** (`shared-fft-packed.patch`) = exact + a new body for `fft()`: for even N, the N real samples are read as
N/2 complex values (no copy), transformed with `fft_impl<false, false>` into `out + 2N`, then split into bins
0..N/2 with the table's exp(-2 pi i k / N); bins N/2+1..N-1 are the conjugate mirror, so the N-complex output
contract is kept. Odd N falls back to the real-input path. `ifft()` is unchanged (bit-identical).

API, header, cache, contracts: no change to `mtmd-audio.h`, `mtmd.h` or `mtmd_audio_cache`; no new table, no
static state, no allocation, no thread. `in` is no longer written to (master used `in + N .. ` as scratch); `out`
needs 2N + 2 max(N/2, 2^p) floats (exact) and 3N + 2 max(N/4, 2^(p-1)) (packed), within the documented 8N.
`fft_impl` now takes `const float * in`.

## Lines (git diff of `mtmd-audio.cpp`, `lean/gen/count.py`)
| patch | code +/- | comment-only +/- | blank +/- | file lines | functions | special cases |
|---|---|---|---|---|---|---|
| exact | +88 / -81 | +13 / -17 | +3 / -14 | 2341 -> 2333 | `dft_impl` removed, 1 lambda (`bitrev_next`) | N == 1, n_odd == 1 (first stage fused) |
| packed | +114 / -82 | +16 / -17 | +4 / -14 | 2341 -> 2362 | same + new `fft()` body | + odd N fallback in `fft()` |
| test (both) | +41 in `tests/test-mtmd-impl.cpp` (`test_audio_fft`, 9 sizes x 2 directions vs a double DFT, rel. error <= 1e-5) | | | | | |
| LEAN (for comparison, `tiny-voca.cpp`) | +83 / -10 | +5 / -1 | | | 1 struct (ctor + 3 methods) | fused size-2/4 stage |
| MID | +100 / -23 | +6 / -1 | | | same | same |

The exact diff is a rewrite of one function (old 130 lines out, new 120 in) rather than an addition: the file
gets 8 lines shorter. ASCII only in the new lines (the removed comments held the only non-ASCII `π`/`·` of this
section), no tabs, no trailing whitespace, 4 spaces, `float * x` style; 0 warnings with g++ 13 in the real build
and 0 with clang++ 18 on the file at the project warning flags (`raw/clang_check.txt`).

## (a) Micro-benchmark, real `libmtmd.so` (`raw/micro_real.txt`)
`shared/harness/micro_real.cpp` dlopens two builds and calls their exported `mtmd_audio_fft` / `mtmd_audio_ifft`
in one process, ABBA, pinned core 2, 41 pairs, ~2 ms of calls per sample, random input. ns per call; ratio =
median ratio [pair-ratio q25, q75].

| caller | dir | N | master ns [q25, q75] | exact ns | exact ratio | packed ns | packed ratio |
|---|---|---|---|---|---|---|---|
| tiny-voca harm | fft | 512 | 9347 [9056, 10023] | 3072 | 0.325 [0.315, 0.336] | 1947 | 0.218 [0.213, 0.224] |
| tiny-voca harm | ifft | 512 | 9221 [8854, 9715] | 3269 | 0.353 [0.344, 0.376] | 3031 | 0.357 [0.352, 0.362] |
| tiny-voca noise | fft | 256 | 4399 [4317, 4551] | 1397 | 0.314 [0.309, 0.325] | 934 | 0.216 [0.213, 0.222] |
| tiny-voca noise | ifft | 256 | 4177 [4079, 4225] | 1421 | 0.344 [0.333, 0.348] | 1405 | 0.345 [0.339, 0.349] |
| s3tok / ve / voxtral-rt | fft | 400 | 20149 [19910, 20815] | 5274 | 0.264 [0.258, 0.274] | 4182 | 0.209 [0.202, 0.213] |
| kaldi fbank | fft | 512 | 8985 [8865, 9213] | 2923 | 0.326 [0.322, 0.336] | 1957 | 0.219 [0.212, 0.224] |
| matcha | fft | 1920 | 64298 [63700, 65714] | 19606 | 0.302 [0.288, 0.316] | 12505 | 0.196 [0.194, 0.206] |
| zonos3 | ifft | 736 | 37297 [36898, 39600] | 10380 | 0.277 [0.266, 0.285] | 10196 | 0.277 [0.272, 0.280] |
| zonos3 | ifft | 2208 | 322212 [320207, 333927] | 71304 | 0.220 [0.217, 0.226] | 71511 | 0.223 [0.218, 0.227] |
| lfm3a istft | ifft | 1280 | 23168 [22923, 23557] | 10375 | 0.446 [0.441, 0.454] | 10553 | 0.447 [0.432, 0.465] |
| soprano istft | ifft | 2048 | 37286 [36780, 37849] | 14360 | 0.385 [0.379, 0.393] | 14440 | 0.379 [0.362, 0.384] |

(Master columns are from the master-vs-exact run; the master-vs-packed run gives the same master times within
5 %.) The inverse rows of packed equal exact (same code). For reference, LEAN's own 512-point real FFT
(`tiny_voca_rfft`) takes 0.8-0.9 us; the shared packed one takes 1.95 us, the shared exact one 3.1 us.

Standalone harness at the same flags (`raw/micro_standalone_O3.txt`) agrees within a few %. With `-march=native`
(standalone, `raw/micro_standalone_native.txt`, old also native): exact 0.15-0.29, packed fwd 0.14-0.20; there
exact is no longer bit-identical (FMA contraction differs with the code shape): worst 4.8e-7 relative.

## (b) tiny-voca synth with the unchanged master `tiny-voca.cpp`
### Real library harness (`raw/real_synth_ab_run2.txt`; run 1 with downloads in the background: `raw/real_synth_ab.txt`)
`lean/harness/real.cpp`: two builds dlopened, exported `mtmd_graph_tiny_voca::synth` called on all 8 dumps
(2868 frames) per sample, 40 ABBA pairs, pinned core. Ratio of medians, pair-ratio median [q25, q75]. Share of
LEAN's gain = (1 - ratio) / (1 - ratio of LEAN).

| build | what changes | ratio vs master | pair median [q25, q75] | share of LEAN's gain | share of MID's gain |
|---|---|---|---|---|---|
| exact | `mtmd-audio.cpp` only (bit-exact) | 0.495 | 0.499 [0.480, 0.512] | 73 % | 69 % |
| packed | `mtmd-audio.cpp` only | 0.440 | 0.448 [0.431, 0.454] | 80 % | 76 % |
| exact_pc | exact + MID's pulse-train hunk in `tiny-voca.cpp` | 0.459 | 0.460 [0.436, 0.500] | 79 % | 74 % |
| packed_pc | packed + MID's pulse-train hunk | 0.399 | 0.409 [0.382, 0.420] | 86 % | 81 % |
| LEAN | `tiny-voca.cpp` only | 0.305 | 0.313 [0.295, 0.318] | 100 % | 95 % |
| MID | `tiny-voca.cpp` only | 0.266 | 0.274 [0.252, 0.283] | | 100 % |
| master2 (control) | none | 0.982 | 1.007 [0.932, 1.076] | | |

Direct pairs: packed / LEAN = 1.42 [1.39, 1.49]; packed_pc / MID = 1.49 [1.43, 1.61]. Run 1 gave the same
picture (exact 0.516, packed 0.448, LEAN 0.312, MID 0.288, packed_pc 0.418, exact_pc 0.457, pair medians).

Why the shared FFT stays behind LEAN for tiny-voca: LEAN builds its plan once per filter call (per-stage
contiguous twiddles, a bit-reversal table, fused first two stages) and also uses a half-size real FFT for the
inverse; the shared one regathers the twiddles of every stage on every call (the cache layout stays as is), has
one fused stage, and its inverse is a full N-point complex transform because `mtmd_audio_ifft` returns N complex
values.

### llama-tts (`llama_tts.txt`, raw `tts/tts_raw.txt`)
`lean/tts/run_tts.sh`: llama-tts with TV_TIMING, 6 builds interleaved with the order rotated every repetition,
median of 10, model `ggml-org/tiny-voca-GGUF` F32 (sha256 in `raw/gguf_sha256.txt`), voices F2/M2, short / medium
/ long text, threads 1 (pinned) and 4. Synth time ratio vs master:

| thr | text | voice | frames | master synth ms | exact | packed | LEAN | MID | packed_pc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | short | F2 | 98 | 14.3 | 0.536 | 0.451 | 0.352 | 0.328 | 0.434 |
| 1 | short | M2 | 89 | 12.6 | 0.499 | 0.458 | 0.328 | 0.283 | 0.477 |
| 1 | medium | F2 | 293 | 44.1 | 0.499 | 0.414 | 0.297 | 0.325 | 0.384 |
| 1 | medium | M2 | 264 | 41.4 | 0.538 | 0.524 | 0.294 | 0.303 | 0.381 |
| 1 | long | F2 | 1022 | 161.4 | 0.445 | 0.413 | 0.300 | 0.246 | 0.385 |
| 1 | long | M2 | 964 | 136.6 | 0.501 | 0.442 | 0.311 | 0.285 | 0.410 |
| 4 | short | F2 | 98 | 14.5 | 0.507 | 0.422 | 0.303 | 0.323 | 0.548 |
| 4 | short | M2 | 89 | 13.4 | 0.473 | 0.481 | 0.294 | 0.325 | 0.385 |
| 4 | medium | F2 | 293 | 41.5 | 0.504 | 0.449 | 0.307 | 0.375 | 0.437 |
| 4 | medium | M2 | 264 | 37.7 | 0.498 | 0.446 | 0.321 | 0.276 | 0.402 |
| 4 | long | F2 | 1022 | 161.9 | 0.446 | 0.401 | 0.275 | 0.245 | 0.384 |
| 4 | long | M2 | 964 | 146.7 | 0.486 | 0.435 | 0.304 | 0.376 | 0.400 |

Ranges over the 12 configurations (synth / net+synth / whole-process wall, vs master): exact 0.45-0.54 /
0.73-0.93 / 0.80-1.04; packed 0.40-0.52 / 0.70-0.90 / 0.82-0.98; LEAN 0.28-0.35 / 0.61-0.94 / 0.74-1.08; MID
0.24-0.38 / 0.63-0.87 / 0.73-1.05; packed_pc 0.38-0.55 / 0.73-0.94 / 0.82-1.02. Example, long F2, 1 thread:
synth 161.4 ms master, 71.8 exact, 66.7 packed, 48.4 LEAN, 39.8 MID, 62.1 packed_pc, on a wall time of 990 ms.
Short texts are dominated by model load (wall ~270 ms) and the wall ratios there are noise. The noisier VM in
this session shows in single cells (e.g. MID 4 threads long M2 0.376); the real-library A/B above is the
reference.

## (c) Other users
### End to end: Soprano-1.1-80M (`soprano/summary.txt`)
`ggml-org/Soprano-1.1-80M-GGUF` Q8_0 + mmproj F16 (uses `mtmd_audio_streaming_istft`, n_fft 2048, hop 512),
llama-tts, long text (10.4 s of 32 kHz audio, 648 frames), 6 interleaved runs per build
(`shared/harness/soprano.sh`). Wall ms, median [min, max]:

| threads | master | exact | packed |
|---|---|---|---|
| 4 | 838 [765, 927] | 814 [742, 1037] (0.97) | 848 [778, 896] (1.01) |
| 1 (pinned) | 2443 [2341, 2541] | 2464 [2343, 2676] (1.01) | 2398 [2304, 2485] (0.98) |

No measurable change: on master the iSTFT costs ~40 us per frame (8.05 ms per 200 frames, below), i.e. ~26 ms of
the 840 ms run; the patch saves ~15 ms of it, below the run-to-run noise. WAVs: exact and packed are byte-identical to master (331776 samples), as is a
second master run.

### Function level, real `libmtmd.so` (`raw/other_users_5rounds.txt`, `raw/other_users_summary.txt`)
`shared/harness/other.cpp`, linked against libmtmd and run with each build's `LD_LIBRARY_PATH`, builds alternated,
5 rounds x 15 calls, pinned core; median over rounds of the per-round median, ms. Input: the master llama-tts WAV
of the long F2 text (24 kHz speech, used as 16 kHz samples), and its STFT for the inverse users.

| function (FFT size) | master ms | exact | packed |
|---|---|---|---|
| `mtmd_audio_preprocessor_chatterbox_ref::preprocess`, 10 s (kaldi 512, s3tok 400 x2, matcha 1920, ve 400; also LUFS, 3/2 upsampler, trim) | 304.5 | 244.1 (0.80) | 230.9 (0.76) |
| `mtmd_audio_log_mel_frame` x500 (400, 128 mels; voxtral realtime) | 18.95 | 11.68 (0.62) | 11.11 (0.59) |
| `mtmd_audio_streaming_istft` 200 frames (1280 / 320, lfm3a) | 5.07 | 2.44 (0.48) | 2.38 (0.47) |
| `mtmd_audio_streaming_istft` 200 frames (2048 / 512, soprano) | 8.05 | 3.45 (0.43) | 3.33 (0.41) |
| `mtmd_audio_zonos3_synth` push + flush, 120 columns (736 + 2208; synthetic mixer / resampler kernels) | 59.09 | 24.61 (0.42) | 24.55 (0.42) |

The zonos3 synth (2.4x) and the two iSTFT shapes (2.1-2.4x) gain the most; in `log_mel_frame` the mel filterbank
(128 x 201 double MACs) is now as large as the FFT; the chatterbox preprocessor has other work besides the FFT.

## Numerical safety
### FFT outputs (`raw/fft_parity_O3.txt`, `raw/fft_parity_native.txt`)
Per size in use: 20 random inputs (uniform real samples for fwd; Hermitian spectra with random log magnitude and
phase for inv, like a vocoder head) + 8 real-audio inputs (Hann frames of speech for fwd, their spectra for inv).
errO / errN = SNR of master / new output against a double-precision DFT.

| N | dir | exact: bit-identical | packed: bit-identical | packed max abs / max rel (vs frame max) | packed SNR vs master | errO / errN (packed) dB |
|---|---|---|---|---|---|---|
| 512 | fft | 28/28 | 0/28 | 9.5e-6 / 3.0e-7 | 135.5 | 137.2 / 137.0 |
| 256 | fft | 28/28 | 0/28 | 5.7e-6 / 2.9e-7 | 136.2 | 137.9 / 137.5 |
| 400 | fft | 28/28 | 0/28 | 9.5e-6 / 3.5e-7 | 133.5 | 135.1 / 134.9 |
| 1920 | fft | 28/28 | 0/28 | 3.1e-5 / 5.1e-7 | 131.3 | 133.5 / 133.4 |
| 512, 256, 736, 2208, 1280, 2048 | ifft | 28/28 | 28/28 | 0 | identical | same |

Sweep (every N in 1..300 plus 384, 400, 480, 512, 640, 736, 768, 960, 1000, 1024, 1280, 1920, 2048, 2208, 3000,
4096; sin/cos table of N and of 2N points; both directions; 1264 cases): exact 1264/1264 bit-identical; packed
932/1264 (all inverse and odd-N forward cases), worst 6.7e-7 relative, worst SNR vs master 125.6 dB. Both are
deterministic (two runs identical). The packed error against the exact DFT is the same as master's (0.1-0.4 dB
lower SNR), so it is a different rounding, not a loss of accuracy.

### Callers
- tiny-voca, llama-tts WAVs (8 utterances, F2/M2, 4 threads) vs master: exact byte-identical on all 8 (and a rerun
  identical); packed same length, at most 1 LSB, SNR 98.9-103.8 dB, the same as LEAN (99.1-104.7) and MID.
- Log-mel / mel features (`raw/other_parity_*.txt`): exact identical for `log_mel_frame`, the chatterbox
  preprocessor (all 5 entries), both iSTFTs and the zonos3 synth. Packed: `log_mel_frame` max abs diff 3.1e-4 in
  log10 units (SNR 117 dB; the largest differences sit in bins ~100 dB below the frame peak), chatterbox entries
  max abs 9.0e-4 on values up to 20.8 (SNR 120 dB); the iSTFT and zonos3 outputs are identical (ifft unchanged).
- Soprano WAV byte-identical for both.
- Expected effect on ASR models: none with exact (bit-identical inputs to the encoders). With packed, only the
  float rounding of the forward real FFT changes (relative 1e-7 level); the mel features move by <= 1e-3 in log
  units, far below what the encoders resolve and below the 0.05 tolerance that `test_audio_log_mel_precision`
  accepts for the cgraph log-mel. Only voxtral realtime and the chatterbox reference features use `fft()` at all.
- `tests/test-mtmd-impl.cpp`: passes on master, exact and packed (294 assertions before, 312 with the new
  `test_audio_fft`; `raw/test_mtmd_impl.txt`). The new test checks `mtmd_audio_fft` / `mtmd_audio_ifft` against a
  double DFT for N = 1, 2, 3, 12, 256, 400, 512, 736, 1920, both directions (relative error <= 1e-5), so it guards
  the odd-part and power-of-two paths of any future change.

## Prior art (re-checked, GitHub search API, 2026-11-22)
- No llama.cpp PR or issue found that speeds up `mtmd_audio_fft` or the mtmd preprocessing FFT (searches:
  `mtmd_audio_fft`, `fft` in titles of PRs/issues, `fft_impl`, `iterative fft`, audio fft optimize/speed). The
  only PR with "fft" in its title is 22770 (whisper tail truncation, a padding fix, merged).
- llama.cpp 31808 "mtmd: move log-mel computation to cgraph" (ngxson, merged 2026-11-07): moves the log-mel of
  most preprocessors to a cgraph DFT matmul to drop thread creation (wasm); "DFT matmul on Metal causes the result
  to deviate a lot", so it uses `ggml_custom` on CPU. ggerganov's comment there (user-provided FFT callback that
  could replace "the simple Cooley-Tukey implementation hardcoded in mtmd-audio.cpp", which stays as fallback) is
  as reported by the previous session; comments are not in the search results and were not re-read.
- llama.cpp 19489 "ggml: add GGML_OP_ISTFT with CPU (PocketFFT) and Vulkan backends": closed, not merged.
- llama.cpp 12514 "llama-tts : precompute irFFT theta" (old llama-tts, ~70 % faster irfft threads): closed, not
  merged. 18645 added `mtmd_audio_streaming_istft` (merged).
- ggml 1105 "ggml : add ggml_fft and ggml_ifft operator": the API still reports it open; the maintainer's
  "not worth adding to ggml for now" is as reported by the previous session.
- whisper.cpp (where this FFT comes from): 1142 precalculated sin/cos tables (merged 2023-08), 2080 fft sizes fix
  (merged 2024-04), 2242 "optimize fft()" (reuse preallocated buffers instead of per-call vectors, mel 125 -> 100 ms
  for 5 min of audio; merged 2024-06), 583 "port fft to pocketfft" (closed, not merged). So the recursive
  Cooley-Tukey has been tuned by allocation and tables, never rewritten iteratively; a dependency (pocketfft) was
  turned down.

## Deviations
- The micro-benchmark main table uses random input only; parity uses random + real audio.
- `mtmd_audio_mel_energies` is static, so it was measured through the chatterbox preprocessor (built from a
  default `mtmd_model` with `n_mel_bins = 80`, `audio_sample_rate = 16000`) and `log_mel_frame`, not on its own.
  The zonos3 synth was measured with synthetic mixer / resampler kernels (no 1.6 GB model download); its FFT work
  is the same as with the real kernels. Voxtral realtime (4B) and chatterbox / zonos3 were not run end to end.
- The tiny-voca dumps (`../dumps`) are the previous session's; the real-audio parity inputs are a master
  llama-tts WAV, so they are TTS speech, not recorded speech.
- `-march=native` numbers are from the standalone harness, not a native llama.cpp build.
- ggerganov's PR comments (31808, ggml 1105) were not re-read (the search API does not return comments).
- The WAV files (llama-tts tiny-voca and Soprano outputs, ~13 MB) are not committed; the parity numbers above
  come from them and they are regenerated by `lean/tts/run_tts.sh` / `shared/harness/soprano.sh`.
- One machine, a noisy shared VM; the control pair and quartiles are given. Run 1 of the synth A/B overlapped with
  model downloads and is kept only as a cross-check.
