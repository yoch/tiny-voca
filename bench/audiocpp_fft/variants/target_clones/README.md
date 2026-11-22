# Variant: function multiversioning (not portable, not for opt/)

`source_filter.cpp` is the final `opt/src/source_filter.cpp` with one addition: `pulse_train()` and `shape()`
carry `__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))` (GCC/Clang, x86-64 ELF
only; elsewhere the macro is empty). The loader picks the AVX-512, AVX2 or baseline clone for the CPU, so a build
without `-march` (which is how audio.cpp compiles this file) still runs the wide code. It breaks the main
candidate's rule "no compiler extension / ISA assumption", hence a variant. Use with the header in `opt/include`.
Modified from audio.cpp 337af0c (Apache-2.0, see LICENSE).

Measured (results/final_timings.txt, results/real_binary.txt), Xeon @ 2.10 GHz with AVX-512, g++ 13 `-O3`:
- harness: 27.26 -> 13.19 us per acoustic frame vs the portable opt/ (paired ratio 0.482); parity vs the
  reference 124.8 dB, 1 LSB, 941 samples differ.
- audio.cpp CLI end to end, 6 cases: 2434.5 -> 2228.4 ms total (ratio to the unchanged binary 0.749 -> 0.694).
The framework FFT (fft.cpp) is not cloned and stays baseline SSE2 here.
