# Shared mtmd FFT: summary and verdict

Question: instead of giving tiny-voca its own FFT (LEAN / MID), can the shared FFT of `tools/mtmd/mtmd-audio.cpp` be
made fast enough, with a small, reviewable, numerically safe patch, that tiny-voca (unchanged) and every other
caller get the speed-up? Target llama.cpp master `8d60873df` (2026-11-22). Machine: 4-vCPU KVM VM, Xeon @ 2.10GHz,
g++ 13.3. Flags of `mtmd-audio.cpp` (own TU): `-O3 -DNDEBUG -fPIC` + project warnings, no `-march`. Details,
tables and method: `ANALYSIS.md`.

## Candidates (`../shared/`)
| | what | `mtmd-audio.cpp` code +/- | comment +/- | outputs vs master |
|---|---|---|---|---|
| **exact** (`shared-fft-exact.patch`) | `dft_impl` + recursive `fft_impl` replaced by one iterative `fft_impl`: odd-size DFT of all 2^p subsequences at once (incremental table index, no `%`), first radix-2 stage fused into the bit-reversed gather, in-place stages with contiguous per-stage twiddles in the scratch | +88 / -81 (file 8 lines shorter) | +13 / -17 | **bit-identical for every N** (1264/1264 sizes x tables x directions; all callers' outputs, all WAVs) |
| **packed** (`shared-fft-packed.patch`) | exact + `fft()` packs N real samples as N/2 complex values and splits (even N) | +114 / -82 | +16 / -17 | ifft identical; fft within 6.7e-7 relative (same error vs a double DFT as master), WAVs <= 1 LSB, log-mel <= 3e-4 |

Both: no change to `mtmd-audio.h`, `mtmd.h`, `mtmd_audio_cache` or the scratch contracts (input no longer written;
`out` stays within 8N), no new table, no static state, no allocation, no thread, no intrinsics, C++17, ASCII,
0 warnings (g++ 13 real build, clang++ 18). Both add `test_audio_fft` (+41 lines) to `tests/test-mtmd-impl.cpp`:
FFT / IFFT vs a double DFT for N = 1, 2, 3, 12, 256, 400, 512, 736, 1920; passes on master and both candidates.

## Callers and sizes
tiny-voca 512 / 256 (2 fwd real + 1 inv complex per STFT frame); `mtmd_audio_log_mel_frame` 400 (voxtral
realtime); `mtmd_audio_mel_energies` 400 (s3tok, voice encoder), 1920 = 128 x 15 (matcha), 512 (kaldi fbank),
all in the chatterbox speaker reference; zonos3 synth ifft 736 = 32 x 23 and 2208 = 32 x 69;
`mtmd_audio_streaming_istft` 1280 = 256 x 5 (LFM3A_GEN) and 2048 (Soprano). The log-mel of whisper, qwen3a,
dots3note, mimo, qwen3tts_spk, conformer, lfm3a, granite, gemma4a, parakeet and the others moved to the cgraph
DFT in PR 31808 and no longer calls `fft()`.

## Prior art
No llama.cpp PR or issue that speeds up `mtmd_audio_fft` was found. 31808 (cgraph log-mel, merged), 19489 (ISTFT op
with PocketFFT, closed), 12514 (irfft theta, closed), ggml 1105 (`ggml_fft`, still open per the API); in
whisper.cpp, where this FFT comes from: 1142 (sin/cos tables) and 2242 (preallocated buffers, mel 125 -> 100 ms)
merged, 583 (pocketfft) not merged. The recursive Cooley-Tukey was never rewritten.

## Numbers (ratio new / master; real `libmtmd.so`, in-process ABBA, pinned core, medians)
| | exact | packed | LEAN | MID |
|---|---|---|---|---|
| FFT micro, 512 fwd / 512 inv | 0.33 / 0.35 | 0.22 / 0.36 | (own rfft: 0.8-0.9 us vs 9.3 us) | |
| FFT micro, 400 / 1920 fwd | 0.26 / 0.30 | 0.21 / 0.20 | | |
| FFT micro, 736 / 2208 / 1280 / 2048 inv | 0.28 / 0.22 / 0.45 / 0.39 | same | | |
| tiny-voca synth (real lib, 8 dumps) | **0.50** [0.48, 0.51] | **0.45** [0.43, 0.45] | 0.31 [0.30, 0.32] | 0.27 [0.25, 0.28] |
| tiny-voca synth, + MID's pulse-train hunk | 0.46 | 0.41 [0.38, 0.42] | | 0.27 |
| share of LEAN's synth gain | 73 % | 80 % (86 % with the pulse train) | 100 % | |
| llama-tts synth, 12 configs (threads 1 / 4) | 0.45-0.54 | 0.40-0.52 | 0.28-0.35 | 0.24-0.38 |
| zonos3 synth (736 + 2208) | 0.42 | 0.42 | | |
| streaming iSTFT 1280 / 2048 | 0.48 / 0.43 | 0.47 / 0.41 | | |
| `log_mel_frame` 400 (voxtral rt) | 0.62 | 0.59 | | |
| chatterbox reference preprocessing (10 s) | 0.80 | 0.76 | | |
| Soprano-1.1-80M llama-tts wall, end to end | 0.97-1.01 (noise) | 0.98-1.01 (noise) | | |

Long F2, 1 thread: tiny-voca synth 161 ms (master), 72 (exact), 67 (packed), 48 (LEAN), 40 (MID), in a 990 ms
process.

## Verdict
**A shared-FFT PR is worth it, as the exact variant, on its own merits, but it does not replace LEAN/MID for
tiny-voca.**

- The exact rewrite makes the shared FFT 2.2-4.5x faster for every size in use, with outputs bit-identical to
  master for every N at the default flags. That removes the usual objection to touching shared numerics: no ASR
  or TTS model's output changes, no reference values move, and a reviewer can check it with a byte comparison.
  It is a same-size rewrite of one function (+88/-81 code lines, file -8 lines), removes `dft_impl` and the `%`
  in its inner loop, and needs no header, cache or contract change. Every caller gains: zonos3 synth 2.4x,
  iSTFT 2.1-2.3x, voxtral realtime log-mel frame 1.6x, chatterbox reference features 1.25x, tiny-voca synth 2.0x.
  End to end the gain is small where the FFT is small (Soprano: not measurable).
- For tiny-voca it gives 73 % of LEAN's synth gain (0.50 vs 0.31). Adding the packed real path (+26 code lines,
  rounding-level changes of the forward FFT) gives 80 %, and with MID's pulse-train hunk 86 % of LEAN's gain
  (81 % of MID's). The rest is plan state the shared function cannot hold without changing `mtmd_audio_cache`
  (precomputed per-stage twiddles and bit-reversal table, fused radix-4) and the full complex inverse that the
  `mtmd_audio_ifft` contract requires.
- They are complementary, not alternatives. Recommended order: (1) the exact shared FFT plus `test_audio_fft`, a
  standalone refactor PR that helps every caller (the easiest of all candidates to review, because it is
  bit-exact); (2) if tiny-voca's synth speed matters, LEAN (or MID) on top, which is still 1.4-1.5x faster than
  the shared path for tiny-voca; or, if maintainers prefer no second FFT in model code, MID's pulse-train hunk
  alone (+17/-13 lines in `tiny-voca.cpp`) on top of the shared FFT, accepting 0.46 instead of 0.27. The packed
  real path is an optional follow-up: 1.5x more on forward real FFTs, but it gives up bit-exactness and so needs the
  parity tables below in its PR.
- Which would a maintainer rather review? The exact shared patch: one internal function, no model code, no
  numerics change, faster for all users, in line with ggerganov's view in 31808 that the hardcoded Cooley-Tukey
  stays as the (now fast) fallback. LEAN adds a second, tiny-voca-only FFT of ~80 lines.

What reviewers would need to see: the caller / size table; the bit-exactness evidence (sweep of every N 1..300 +
the sizes in use, tables N and 2N, both directions; tiny-voca and Soprano WAVs byte-identical; log-mel and
chatterbox features identical); the micro table per size and the end-to-end numbers for tiny-voca and zonos3 /
iSTFT; the scratch-contract argument (input no longer written, out within 8N); `test_audio_fft` passing on master
and on the patch. For packed additionally: max relative error 6.7e-7, SNR vs master >= 125 dB, same error as
master against a double DFT, log-mel <= 3e-4 and chatterbox features <= 9e-4 absolute.

## Deviations
- Other users end to end: only Soprano (small public model). Voxtral realtime (4B), chatterbox and zonos3 were
  measured at function level on the real library (zonos3 with synthetic mixer/resampler kernels).
- With `-march=native` the exact variant is no longer bit-identical (FMA contraction depends on the code shape;
  worst 4.8e-7 relative); bit-exactness is a property of the default build.
- Comments of maintainers on 31808 / ggml 1105 were taken from the previous session, not re-read.
- Noisy shared VM; every ratio is interleaved, quartiles and a master-vs-master control are in `ANALYSIS.md`.
