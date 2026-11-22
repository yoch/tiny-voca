#pragma once
// Benchmark stub: only the model settings the waveform head reads (audio.cpp assets.h, TinyVocaConfig).
#include <cstdint>
#include <string>
#include <vector>
namespace engine::models::tiny_voca {
struct TinyVocaConfig {
    int64_t sample_rate = 24000;
    int64_t hop = 256;
    int64_t channels = 192;
    int64_t synth_fft = 512;
    int64_t synth_hop = 128;
    int64_t noise_fft = 256;
    int64_t noise_hop = 64;
    int64_t cep_causal = 80;
    int64_t cep_anti = 16;
    int64_t noise_cep = 32;
    float harmonic_limit = 0.47F;
};
}  // namespace engine::models::tiny_voca
