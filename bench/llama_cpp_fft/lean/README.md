# Lean candidates for the tiny-voca host synth in llama.cpp

Patches against llama.cpp master `e300919cc704ddde3a8a8346a8714f57eeeee71b` (2026-11-22), one file each,
`tools/mtmd/models/tiny-voca.cpp` only (`mtmd-audio.cpp` / `mtmd-audio.h` untouched). Results and the
recommendation: `../results_lean/SUMMARY.md` and `../results_lean/ABLATION.md`.

| patch | what | diff (git, incl. comments/blank) |
|---|---|---|
| `lean.patch` | the 3 FFTs per STFT frame use a packed real FFT (`tiny_voca_rfft`: iterative radix-2 complex FFT of n/2 points, first two stages fused, stack scratch) instead of `mtmd_audio_fft` / `mtmd_audio_ifft` | +93/-11 |
| `mid.patch` | lean + the pulse train of variant C (one exp per frame, phase in cycles, two sin/cos pairs per sample) | +111/-24 |
| `full.patch` | mid + filter response once per envelope (the gain is added to the log magnitude) | +132/-36 |

Each patch applies on its own to master (`git apply --check` clean on `e300919cc`).
`src/<name>.cpp` is the whole patched `tiny-voca.cpp` for reading; `src/ref.cpp` is the master file, `src/C.cpp`
the file with `tiny-voca-rfft-v2.patch` (variant C). The other files in `src/` are the ablation variants.

## Tools
- `gen/mk.py ref.cpp C.cpp A_B_C`: composes the single-idea ablation variants from the master file by exact text
  replacement (`Pw`, `Pt`, `Pc`, `T`, `N`, `R`, `H`, `I`, `J`, see ABLATION.md); `gen/count.py ref.cpp cand.cpp...`:
  changed lines (code / comment-only / blank) as git counts them.
- `harness/`: standalone harness. `build.sh OUT "FLAGS" src/*.cpp` cuts the host-synth section of each candidate file
  (`wrap.sh`), compiles each in its own TU at `-O3 -DNDEBUG -fPIC` (+ FLAGS) with a verbatim copy of the
  mtmd-audio FFT in another TU, all in one shared lib; `OUT/bench parity dumps/u*.bin` (every candidate vs ref on
  the 8 dumps + 1..3-frame cuts: length, SNR, 16-bit max diff, determinism) and `OUT/bench ab A B iters dumps...`
  (interleaved ABBA pairs, whole workload per sample, ratio of medians and pair-ratio quartiles).
- `harness/real.cpp`: the same A/B on the REAL compiled code: dlopens two llama.cpp `build/bin` copies and calls the
  exported `mtmd_graph_tiny_voca::synth` of each `libmtmd.so`. This is the reference measurement (see ABLATION.md for
  why the standalone harness alone is not enough). `harness/ablation_real.sh` rebuilds libmtmd per variant.
- `tts/run_tts.sh OUT BINS GGUF REPS ref b1 b2...`: interleaved llama-tts runs (TV_TIMING, order rotated every
  repetition, threads 1 pinned and 4), then WAVs of 8 utterances per build; `tts/analyze.py OUT ref b1...` prints
  the medians and the WAV parity.

## License
The patches and the files in `src/` and `harness/fft.cpp` contain code from llama.cpp
(https://github.com/ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml authors: see
`../LICENSE-llama.cpp`.

## Reproduce (summary)
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout e300919cc
    git apply ../bench/llama_cpp_fft/timing-instrumentation.patch      # TV_TIMING, measurement only
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-tts -j
    # per candidate: copy lean/src/<name>.cpp to tools/mtmd/models/tiny-voca.cpp, rebuild, copy build/bin to bins/<name>
    # (harness/rebuild_real.sh), then:
    g++ -O2 -std=c++17 harness/real.cpp -o real -ldl      # the synth symbol is baked in from nm -D libmtmd.so
    LD_LIBRARY_PATH=bins/ref taskset -c 2 ./real ab bins/ref bins/lean 40 ../dumps/u*.bin
    harness/build.sh /tmp/h "" src/*.cpp && /tmp/h/bench parity ../dumps/u*.bin
    tts/run_tts.sh OUT bins GGUF_DIR 10 ref lean mid full C && python3 tts/analyze.py OUT ref lean mid full C
Paths inside `harness/rebuild_real.sh` and `harness/ablation_real.sh` point to `/tmp/tv` (the clone, `bins/`).
