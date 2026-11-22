// A/B of the real compiled synth: each build's libmtmd.so is dlopen'ed and mtmd_graph_tiny_voca::synth called directly
//   real parity LIBDIR_REF LIBDIR... -- dumps...        real ab LIBDIR_A LIBDIR_B iters dumps...
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
typedef void (*synth_fn)(const std::vector<float> &, const std::vector<float> &, std::mt19937 &, std::vector<float> &);
static const int n_voc = 134;
struct utt { std::vector<float> voc, f0; };
static synth_fn open_lib(const char * dir) {
    std::string p = std::string(dir) + "/libmtmd.so";
    void * h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); exit(1); }
    synth_fn f = (synth_fn) dlsym(h, "_ZN20mtmd_graph_tiny_voca5synthERKSt6vectorIfSaIfEES4_RSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EERS2_");
    if (!f) { fprintf(stderr, "no synth in %s\n", p.c_str()); exit(1); }
    return f;
}
static std::vector<utt> load(int argc, char ** argv, int first) {
    std::vector<utt> u;
    for (int a = first; a < argc; a++) {
        FILE * fp = fopen(argv[a], "rb"); int64_t nf = 0;
        if (!fp || fread(&nf, 8, 1, fp) != 1) exit(1);
        utt x; x.f0.resize(nf); x.voc.resize(nf * n_voc);
        if (fread(x.f0.data(), 4, nf, fp) != (size_t) nf || fread(x.voc.data(), 4, x.voc.size(), fp) != x.voc.size()) exit(1);
        fclose(fp); u.push_back(x);
    }
    return u;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double run_all(synth_fn f, const std::vector<utt> & us) {
    const double t0 = now_ms();
    for (auto & u : us) { std::mt19937 rng(7); std::vector<float> out; f(u.voc, u.f0, rng, out); }
    return now_ms() - t0;
}
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); double i = p * (v.size() - 1); size_t lo = (size_t) i; return lo + 1 < v.size() ? v[lo] + (i - lo) * (v[lo + 1] - v[lo]) : v[lo]; }
int main(int argc, char ** argv) {
    std::string mode = argv[1];
    if (mode == "ab") {
        synth_fn fa = open_lib(argv[2]), fb = open_lib(argv[3]);
        int iters = atoi(argv[4]);
        auto us = load(argc, argv, 5);
        run_all(fa, us); run_all(fb, us);
        std::vector<double> ta, tb, r;
        for (int i = 0; i < iters; i++) {
            double a, b;
            if (i % 2 == 0) { a = run_all(fa, us); b = run_all(fb, us); } else { b = run_all(fb, us); a = run_all(fa, us); }
            ta.push_back(a); tb.push_back(b); r.push_back(b / a);
        }
        printf("real-ab %s %s iters=%d | A med=%.3f ms | B med=%.3f ms | ratio_of_medians=%.4f | pair_ratio q25=%.4f med=%.4f q75=%.4f\n",
               argv[2], argv[3], iters, pct(ta, 0.5), pct(tb, 0.5), pct(tb, 0.5) / pct(ta, 0.5), pct(r, 0.25), pct(r, 0.5), pct(r, 0.75));
    }
    return 0;
}
