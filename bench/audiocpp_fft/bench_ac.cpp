// audio.cpp tiny-voca waveform head: timing and parity on real dumps (dump.py). Links ONE implementation of
// TinyVocaSourceFilter (reference = audio.cpp 337af0c verbatim, or the optimized copy).
// usage: bench_ac DUMPS RUNS [OUT_DIR]
#include "engine/community_models/tiny_voca/source_filter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace engine::models::tiny_voca;
static std::vector<float> load(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    std::vector<float> v(static_cast<size_t>(f.tellg()) / 4); f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), v.size() * 4); return v;
}
int main(int argc, char ** argv) {
    const std::string d = argv[1]; const int runs = std::stoi(argv[2]); const std::string out = argc > 3 ? argv[3] : "";
    TinyVocaConfig c;
    TinyVocaSourceFilter sf(c, load(d + "/hw.f32"), load(d + "/hb.f32"), load(d + "/nw.f32"), load(d + "/nb.f32"));
    std::ifstream cases(d + "/cases.txt"); std::string name; int64_t frames;
    double total_ms = 0, total_pulse = 0; int64_t total_frames = 0;
    while (cases >> name >> frames) {
        auto hidden = load(d + "/" + name + ".hidden"), f0 = load(d + "/" + name + ".f0"), noise = load(d + "/" + name + ".noise"), ref = load(d + "/" + name + ".wave");
        std::vector<double> times, ptimes; std::vector<float> wave;
        for (int r = 0; r < runs; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            wave = sf.synthesize(hidden, f0, noise, frames);
            auto t1 = std::chrono::steady_clock::now();
            auto p = sf.pulse_train(f0);
            auto t2 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            ptimes.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(times.begin(), times.end()); std::sort(ptimes.begin(), ptimes.end());
        double num = 0, den = 0;
        for (size_t i = 0; i < wave.size(); ++i) { num += double(ref[i]) * ref[i]; den += double(wave[i] - ref[i]) * (wave[i] - ref[i]); }
        printf("%-9s %4lld frames  synth %8.3f ms  (pulse_train %7.3f ms)  SNR vs PyTorch %6.1f dB\n", name.c_str(), (long long)frames,
               times[times.size() / 2], ptimes[ptimes.size() / 2], 10 * std::log10(num / std::max(den, 1e-30)));
        total_ms += times[times.size() / 2]; total_pulse += ptimes[ptimes.size() / 2]; total_frames += frames;
        if (!out.empty()) { std::ofstream o(out + "/" + name + ".f32", std::ios::binary); o.write(reinterpret_cast<const char *>(wave.data()), wave.size() * 4); }
    }
    printf("TOTAL %lld frames: synth %.2f ms (%.1f us/frame), pulse_train %.2f ms\n", (long long)total_frames, total_ms, 1000 * total_ms / total_frames, total_pulse);
}
