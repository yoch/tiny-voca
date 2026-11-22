# audio.cpp tiny-voca waveform head: optimization kit

Target: `src/community_models/tiny_voca/source_filter.cpp` of [audio.cpp](https://github.com/0xShug0/audio.cpp)
@ 337af0c (Apache-2.0). The waveform head is host DSP: a pulse train and noise, each filtered frame by frame by a
cepstral envelope in the STFT domain (harmonic: FFT 512 / hop 128, 96 coefficients; noise: FFT 256 / hop 64,
32 coefficients; acoustic hop 256 at 24 kHz).

## Files
- `bench_ac.cpp`: harness, links ONE implementation of `TinyVocaSourceFilter`; times `synthesize()` and
  `pulse_train()` on real dumps (median of N runs), checks against the PyTorch wave, writes the outputs.
- `stub/`: the only part of `assets.h` the head reads (`TinyVocaConfig`), so nothing else of audio.cpp is built.
- `opt/`: optimized copy of `source_filter.h/.cpp` (modified from audio.cpp, Apache-2.0, see `opt/LICENSE`).
- `dump.py`: real inputs of the head (hidden states, f0, noise) + PyTorch wave + head weights, from the public
  English model `n4ze3m/tiny-voca` run by the upstream code (front end `own`).
- `compare.py`: parity of two output sets (SNR, 16-bit difference after one shared peak scaling).
- `run.sh`: setup (audio.cpp clone, venv, dumps), builds, `results/baseline.txt`.

## What is already done in opt/ (vs reference, on an i7-3770, single-threaded FFT)
Reference cost: ~372 us per acoustic frame. The response of EVERY filter frame was a direct DFT of the cepstrum in
double (coefficients x bins table) plus exp, cos and sin in double per bin.
1. Log response = framework real FFT of the liftered cepstrum, once per acoustic frame, without its gain; harmonic
   interpolation = mean of two log responses (linear transform); gain added to the log magnitude before the same
   clamp [-14, 6]; float.
2. Phase cos/sin once per envelope (the 4 noise filter frames of an acoustic frame share it).
3. Linear head looped along the outputs with transposed weights: same summation order (bit-identical), vectorizes.
4. Pulse train: one exp per frame (geometric series), phase in cycles in [0, 1), two sin/cos pairs per sample.
Result: ~100 us per frame (-73 %); parity SNR 125-130 dB, max 1 LSB on < 0.1 % of 16-bit samples.
Remaining profile: pulse train ~24 us/frame (2 double sincos per sample), shape() ~76 us/frame.
Their FFT default `threads = 0` uses a thread pool per block: a net loss here (measured both ways).

## Constraints for any further change
- Parity vs the reference on every dump: same length, SNR >= 110 dB, 16-bit max difference <= 1 LSB.
- Deterministic; C++17; no new dependency; keep the framework FFT (no hand-written FFT); memory bounded by blocks as
  now; ASCII; audio.cpp comment style (plain sentences explaining why).
- Portable code only in the main candidate: no intrinsics, no `-ffast-math`, no `-march`-specific assumptions.
  Ideas that need them are measured as separate variants.
