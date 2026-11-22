// minimal stand-ins for the llama.cpp declarations used by the mtmd-audio FFT section
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "assert %s\n", #x); abort(); } } while (0)
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    void fill_sin_cos_table(uint32_t n) {   // verbatim from mtmd-audio.cpp
        sin_vals.resize(n);
        cos_vals.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            double theta = (2 * M_PI * i) / n;
            sin_vals[i]  = sinf(theta);
            cos_vals[i]  = cosf(theta);
        }
    }
};
typedef void (*fft_fn)(const mtmd_audio_cache &, float *, int, float *);
