# Shared mtmd FFT candidates

Patches against llama.cpp master `8d60873dfb8619f2b5ed31adfbab07cccb40db5c` (2026-11-22), `tools/mtmd/mtmd-audio.cpp`
plus a new test case in `tests/test-mtmd-impl.cpp`. Model code (`tiny-voca.cpp` etc.) is unchanged. Results and
verdict: `../results_shared/SUMMARY.md`, `../results_shared/ANALYSIS.md`.

| patch | what |
|---|---|
| `shared-fft-exact.patch` | iterative `fft_impl` (batched odd-size DFT, fused first stage, contiguous per-stage twiddles), bit-identical to master |
| `shared-fft-packed.patch` | exact + packed real-input path in `fft()` (N real -> N/2 complex + split) |

`src/mtmd-audio.{master,exact,packed}.cpp` are the whole files. `git apply --check` is clean on `8d60873df`.

## Tools (`harness/`)
- `variants/*.inc`: the FFT section of `mtmd-audio.cpp` for each development step (`old` = master verbatim, `v1`..`v9`,
  `exact`, `packed`); `build.sh OUT "FLAGS" v...` compiles each in its own TU at the real flags (+ FLAGS) with
  `tu.cpp` / `shim.h`; `micro parity [wav]`, `micro sweep`, `micro bench A B iters` (`micro.cpp`).
- `micro_real.cpp`: in-process A/B of the exported `mtmd_audio_fft` / `mtmd_audio_ifft` of two real `libmtmd.so`.
- `other.cpp`: function-level timing and outputs of the other callers (chatterbox reference preprocessor,
  `mtmd_audio_log_mel_frame`, `mtmd_audio_streaming_istft`, `mtmd_audio_zonos3_synth`) on a real build;
  `other cmp A.bin B.bin` compares outputs.
- `rebuild_real.sh name audio tv`: rebuild the clone with `src/mtmd-audio.<audio>.cpp` and `../lean/src/<tv>.cpp`.
- `soprano.sh`: interleaved llama-tts runs of Soprano-1.1-80M and its WAVs.
- The tiny-voca synth A/B and llama-tts runs reuse `../lean/harness/real.cpp` and `../lean/tts/run_tts.sh` /
  `analyze.py`.

## License
The patches, `src/` and `harness/variants/` contain code from llama.cpp (https://github.com/ggml-org/llama.cpp),
MIT License, Copyright (c) 2023-2026 The ggml authors: see `../LICENSE-llama.cpp`.
