// in-process A/B of the exported mtmd_audio_fft / mtmd_audio_ifft of two real libmtmd.so builds (dlopen, RTLD_LOCAL)
//   micro_real LIBDIR_A LIBDIR_B iters
#include "mtmd-audio.h"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
typedef void (*fft_fn)(const mtmd_audio_cache &, float *, int, float *);
struct lib { fft_fn fft, ifft; };
static lib open_lib(const char * dir) {
    std::string p = std::string(dir) + "/libmtmd.so";
    void * h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); exit(1); }
    lib l;
    l.fft  = (fft_fn) dlsym(h, "_Z14mtmd_audio_fftRK16mtmd_audio_cachePfiS2_");
    l.ifft = (fft_fn) dlsym(h, "_Z15mtmd_audio_ifftRK16mtmd_audio_cachePfiS2_");
    if (!l.fft || !l.ifft) { fprintf(stderr, "missing symbols in %s\n", p.c_str()); exit(1); }
    return l;
}
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); double i = p * (v.size() - 1); size_t lo = (size_t) i; return lo + 1 < v.size() ? v[lo] + (i - lo) * (v[lo + 1] - v[lo]) : v[lo]; }
int main(int argc, char ** argv) {
    lib a = open_lib(argv[1]), b = open_lib(argv[2]);
    const int iters = atoi(argv[3]);
    struct fcase { const char * who; int N; bool inv; };
    const fcase cases[] = {
        { "tiny-voca harm fft", 512, false }, { "tiny-voca harm ifft", 512, true }, { "tiny-voca noise fft", 256, false }, { "tiny-voca noise ifft", 256, true },
        { "s3tok/ve/voxtral-rt 400", 400, false }, { "kaldi fbank 512", 512, false }, { "matcha 1920", 1920, false },
        { "zonos3 ifft 736", 736, true }, { "zonos3 ifft 2208", 2208, true }, { "lfm3a istft 1280", 1280, true }, { "soprano istft 2048", 2048, true },
    };
    printf("%-26s %-5s %5s | %9s %9s %9s | %9s %9s %9s | %6s %6s %6s %6s\n", "case", "dir", "N", "A_med_ns", "A_q25", "A_q75", "B_med_ns", "B_q25", "B_q75", "ratio", "pr_q25", "pr_med", "pr_q75");
    for (const auto & c : cases) {
        mtmd_audio_cache cache;
        cache.fill_sin_cos_table(c.N);   // from the linked library, the same code in every build
        std::mt19937 rng(5); std::uniform_real_distribution<float> u(-1, 1);
        std::vector<float> in(8 * c.N, 0.0f), out(8 * c.N);
        for (int i = 0; i < (c.inv ? 2 * c.N : c.N); i++) in[i] = u(rng);
        fft_fn fa = c.inv ? a.ifft : a.fft, fb = c.inv ? b.ifft : b.fft;
        double t0 = now_ns(); int k0 = 0; while (now_ns() - t0 < 2e6) { fa(cache, in.data(), c.N, out.data()); k0++; }
        const int K = std::max(5, k0);
        auto sample = [&](fft_fn f) { double s = now_ns(); for (int k = 0; k < K; k++) f(cache, in.data(), c.N, out.data()); return (now_ns() - s) / K; };
        sample(fa); sample(fb);
        std::vector<double> ta, tb, r;
        for (int i = 0; i < iters; i++) {
            double x, y;
            if (i % 2 == 0) { x = sample(fa); y = sample(fb); } else { y = sample(fb); x = sample(fa); }
            ta.push_back(x); tb.push_back(y); r.push_back(y / x);
        }
        printf("%-26s %-5s %5d | %9.0f %9.0f %9.0f | %9.0f %9.0f %9.0f | %6.3f %6.3f %6.3f %6.3f\n", c.who, c.inv ? "ifft" : "fft", c.N,
               pct(ta, .5), pct(ta, .25), pct(ta, .75), pct(tb, .5), pct(tb, .25), pct(tb, .75), pct(tb, .5) / pct(ta, .5), pct(r, .25), pct(r, .5), pct(r, .75));
    }
    printf("A=%s B=%s iters=%d, ns per call (real libmtmd.so, in-process, ABBA)\n", argv[1], argv[2], iters);
    return 0;
}
