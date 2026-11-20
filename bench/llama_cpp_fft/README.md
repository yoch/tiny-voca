# llama.cpp tiny-voca host synth benchmark kit

Reruns, on any Linux machine, the measurements of the packed real-FFT patch for the tiny-voca
source-filter synthesizer in llama.cpp (`tools/mtmd/models/tiny-voca.cpp`), against llama.cpp
`9d6f04f` (2026-11-20).

    bash bench/llama_cpp_fft/run.sh

- `tiny-voca-rfft-v2.patch`: the synth change being measured (the "fast" build).
- `tts-threads-9d6f04f.patch`: makes `llama-tts -t N` reach the mtmd context, so thread counts are controlled (both builds).
- `timing-instrumentation.patch`: `TV_TIMING` hooks that print the stack and synth times (both builds, measurement only).
- `bench.cpp` + `dumps/*.bin`: standalone harness (reference copy of the old synth vs variants), fed with
  `voc`/`f0` dumped from llama-tts for 8 English utterances (F2/M2).

Results go to `bench/llama_cpp_fft/results/<host>-<utc>/`. First run: Intel i7-3770 (AVX, no AVX2),
synth -74 to -75 %, stacks + synth -31 to -40 %, 16-bit WAV within 1 LSB.

## License
The patches and `bench.cpp` contain code from llama.cpp (https://github.com/ggml-org/llama.cpp), MIT License,
Copyright (c) 2023-2026 The ggml authors: see `LICENSE-llama.cpp`. The rest of this repository is Apache-2.0.
