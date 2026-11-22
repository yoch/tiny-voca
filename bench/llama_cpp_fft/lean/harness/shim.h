// minimal stand-ins for the llama.cpp declarations used by the tiny-voca host synth
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "assert %s\n", #x); abort(); } } while (0)

struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> hann_window;
    void fill_sin_cos_table(uint32_t n);
    void fill_hann_window(uint32_t length, bool periodic);
};
void mtmd_audio_fft(const mtmd_audio_cache & cache, float * in, int N, float * out);
void mtmd_audio_ifft(const mtmd_audio_cache & cache, float * in, int N, float * out);

struct tiny_voca_consts {
    static constexpr int   hop              = 256;
    static constexpr int   n_cep_causal     = 80;
    static constexpr int   n_cep_anti       = 16;
    static constexpr int   n_noise_cep      = 32;
    static constexpr int   harm_fft         = 512;
    static constexpr int   harm_hop         = 128;
    static constexpr int   harm_steps       = hop / harm_hop;
    static constexpr int   noise_fft        = 256;
    static constexpr int   noise_hop        = 64;
    static constexpr int   noise_steps      = hop / noise_hop;
    static constexpr int   n_voc            = n_cep_causal + n_cep_anti + harm_steps + n_noise_cep + noise_steps;
    static constexpr float sample_rate      = 24000.0f;
    static constexpr float harmonic_limit   = 0.47f;
};
typedef void (*synth_fn)(const std::vector<float> &, const std::vector<float> &, std::mt19937 &, std::vector<float> &);
