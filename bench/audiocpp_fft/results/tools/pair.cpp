// In-process paired timing of two TinyVocaSourceFilter builds (namespaces tv_a and tv_b, see pair.sh):
// every iteration runs each case with A then B (order swapped every other iteration), so both see the same
// machine state. usage: pair DUMPS ITERS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#define tiny_voca tv_a
#include "hdr_a.h"
#undef tiny_voca
#define tiny_voca tv_b
#include "hdr_b.h"
#undef tiny_voca
static std::vector<float> load(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    std::vector<float> v(static_cast<size_t>(f.tellg()) / 4); f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), v.size() * 4); return v;
}
struct Case { std::string name; int64_t frames; std::vector<float> hidden, f0, noise; };
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char ** argv) {
    const std::string d = argv[1]; const int iters = std::stoi(argv[2]);
    engine::models::tv_a::TinyVocaSourceFilter A({}, load(d + "/hw.f32"), load(d + "/hb.f32"), load(d + "/nw.f32"), load(d + "/nb.f32"));
    engine::models::tv_b::TinyVocaSourceFilter B({}, load(d + "/hw.f32"), load(d + "/hb.f32"), load(d + "/nw.f32"), load(d + "/nb.f32"));
    std::vector<Case> cases; std::ifstream cf(d + "/cases.txt"); std::string name; int64_t frames, total_frames = 0;
    while (cf >> name >> frames) { cases.push_back({name, frames, load(d + "/" + name + ".hidden"), load(d + "/" + name + ".f0"), load(d + "/" + name + ".noise")}); total_frames += frames; }
    if (argc > 3 && std::string(argv[3]) == "pulse") {  // accuracy of B's pulse train against A's
        for (auto & c : cases) {
            auto pa = A.pulse_train(c.f0), pb = B.pulse_train(c.f0); double num = 0, den = 0, mx = 0;
            for (size_t i = 0; i < pa.size(); ++i) { num += double(pa[i]) * pa[i]; den += double(pb[i] - pa[i]) * (pb[i] - pa[i]); mx = std::max(mx, (double)std::fabs(pb[i] - pa[i])); }
            printf("%-9s pulse SNR %6.1f dB  max abs diff %.3g  (peak %.3g)\n", c.name.c_str(), 10 * std::log10(num / std::max(den, 1e-300)), mx, (double)*std::max_element(pa.begin(), pa.end()));
        }
        return 0;
    }
    if (argc > 3 && std::string(argv[3]) == "edge") {  // B against A on the start of the longest case cut to N frames
        const Case & c = cases.back(); double worst = 1e9; int64_t maxlsb = 0;
        for (int64_t n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 34, 63, 64, 65, 66, 127, 128, 129, 200, 590}) {
            std::vector<float> h(c.hidden.begin(), c.hidden.begin() + n * 192), f(c.f0.begin(), c.f0.begin() + n), z(c.noise.begin(), c.noise.begin() + n * 256);
            auto wa = A.synthesize(h, f, z, n), wb = B.synthesize(h, f, z, n); double num = 0, den = 0, peak = 0;
            if (wa.size() != wb.size()) { printf("N=%lld size mismatch\n", (long long)n); return 1; }
            for (size_t i = 0; i < wa.size(); ++i) { num += double(wa[i]) * wa[i]; den += double(wb[i] - wa[i]) * (wb[i] - wa[i]); peak = std::max(peak, (double)std::fabs(wa[i])); }
            int64_t lsb = 0; const double sc = 0.95 / peak * 32767;
            for (size_t i = 0; i < wa.size(); ++i) lsb = std::max<int64_t>(lsb, std::llabs(std::llround(wa[i] * sc) - std::llround(wb[i] * sc)));
            const double snr = 10 * std::log10(num / std::max(den, 1e-300)); worst = std::min(worst, snr); maxlsb = std::max(maxlsb, lsb);
            printf("N=%-4lld samples %6zu  SNR %6.1f dB  16-bit max diff %lld\n", (long long)n, wa.size(), snr, (long long)lsb);
        }
        printf("EDGE worst SNR %.1f dB, max diff %lld LSB\n", worst, (long long)maxlsb);
        return 0;
    }
    std::vector<std::vector<double>> ta(cases.size()), tb(cases.size()); std::vector<double> sa, sb, ratio;
    double sink = 0;
    auto run = [&](auto & sf, const Case & c) {
        auto t0 = std::chrono::steady_clock::now();
        auto w = sf.synthesize(c.hidden, c.f0, c.noise, c.frames);
        auto t1 = std::chrono::steady_clock::now(); sink += w[w.size() / 2];
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    for (int it = -2; it < iters; ++it) {
        double xa = 0, xb = 0;
        for (size_t i = 0; i < cases.size(); ++i) {
            double a, b;
            if (it & 1) { b = run(B, cases[i]); a = run(A, cases[i]); } else { a = run(A, cases[i]); b = run(B, cases[i]); }
            if (it >= 0) { ta[i].push_back(a); tb[i].push_back(b); }
            xa += a; xb += b;
        }
        if (it >= 0) { sa.push_back(xa); sb.push_back(xb); ratio.push_back(xb / xa); }
    }
    for (size_t i = 0; i < cases.size(); ++i)
        printf("%-9s %4lld frames  A %8.3f ms  B %8.3f ms  B/A %.3f\n", cases[i].name.c_str(), (long long)cases[i].frames, med(ta[i]), med(tb[i]), med(tb[i]) / med(ta[i]));
    std::sort(ratio.begin(), ratio.end());
    printf("TOTAL %lld frames: A %.2f us/frame  B %.2f us/frame  paired B/A median %.3f (quartiles %.3f .. %.3f)  [%g]\n", (long long)total_frames,
           1000 * med(sa) / total_frames, 1000 * med(sb) / total_frames, ratio[ratio.size() / 2], ratio[ratio.size() / 4], ratio[3 * ratio.size() / 4], sink * 0);
}
