// FFT micro-benchmark and parity of mtmd-audio FFT variants (each variant in its own TU, see tu.cpp, build.sh)
//   micro parity [wav]            every variant vs old, every case: bit-exact frames, max abs / rel diff, SNR, error vs double DFT
//   micro bench A B [iters]       interleaved ABBA A/B per case, per-call median and quartiles, pair-ratio quartiles
#include "shim.h"
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <string>
struct variant { const char * name; fft_fn fft, ifft; };
extern variant g_variants[];
extern int g_n_variants;

struct fcase { const char * who; int N; bool inv; };
static const fcase g_cases[] = {
    { "tiny-voca harm fft",  512, false }, { "tiny-voca harm ifft",  512, true },
    { "tiny-voca noise fft", 256, false }, { "tiny-voca noise ifft", 256, true },
    { "s3tok/ve/voxtral-rt mel 400", 400, false }, { "kaldi fbank 512", 512, false },
    { "matcha mel 1920", 1920, false },
    { "zonos3 ifft 736", 736, true }, { "zonos3 ifft 2208", 2208, true },
    { "lfm3a istft 1280", 1280, true }, { "istft 2048 (soprano?)", 2048, true },
    { "edge 1", 1, false }, { "edge 1 inv", 1, true }, { "edge 2", 2, false }, { "edge 2 inv", 2, true }, { "edge 3", 3, true }, { "edge 6", 6, false }, { "edge 12 inv", 12, true },
};

static const variant & find(const std::string & n) {
    for (int i = 0; i < g_n_variants; i++) if (n == g_variants[i].name) return g_variants[i];
    fprintf(stderr, "no variant %s\n", n.c_str()); exit(1);
}
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); double i = p * (v.size() - 1); size_t lo = (size_t) i; return lo + 1 < v.size() ? v[lo] + (i - lo) * (v[lo + 1] - v[lo]) : v[lo]; }

// input frames for a case: real windowed frames (fwd) or the half spectrum of one, mirrored (inv), as the callers build them
static std::vector<std::vector<double>> make_inputs(const fcase & c, const std::vector<float> & wav, int n_rand, uint32_t seed) {
    std::vector<std::vector<double>> v;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int N = c.N;
    auto spec_of = [&](const std::vector<double> & x) {
        std::vector<double> s(2 * N);
        for (int k = 0; k < N; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++) { double t = -2 * M_PI * (double) ((int64_t) k * n % N) / N; re += x[n] * cos(t); im += x[n] * sin(t); }
            s[2 * k] = re; s[2 * k + 1] = im;
        }
        return s;
    };
    // random
    for (int i = 0; i < n_rand; i++) {
        if (!c.inv) { std::vector<double> x(N); for (auto & e : x) e = u(rng); v.push_back(x); }
        else {
            // hermitian half spectrum, random log magnitude / phase like a vocoder head, imag of DC/nyquist 0
            std::vector<double> s(2 * N, 0.0);
            for (int k = 0; k <= N / 2; k++) { double a = exp(3 * u(rng)), ph = M_PI * u(rng); s[2 * k] = a * cos(ph); s[2 * k + 1] = (k == 0 || k == N / 2) ? 0 : a * sin(ph); }
            for (int k = 1; k < N / 2; k++) { s[2 * (N - k)] = s[2 * k]; s[2 * (N - k) + 1] = -s[2 * k + 1]; }
            v.push_back(s);
        }
    }
    // real audio: hann windowed frames at a few offsets
    if (!wav.empty() && (int) wav.size() > N) {
        const int n_real = 8;
        for (int i = 0; i < n_real; i++) {
            size_t off = (size_t) ((wav.size() - N) * (i + 0.5) / n_real);
            std::vector<double> x(N);
            for (int n = 0; n < N; n++) x[n] = wav[off + n] * (0.5 - 0.5 * cos(2 * M_PI * n / N));
            v.push_back(c.inv ? spec_of(x) : x);
        }
    }
    return v;
}
static std::vector<double> dft_ref(const std::vector<double> & in, int N, bool inv) {
    std::vector<double> out(2 * N);
    for (int k = 0; k < N; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double t = (inv ? 2 : -2) * M_PI * (double) ((int64_t) k * n % N) / N;
            double a = inv ? in[2 * n] : in[n], b = inv ? in[2 * n + 1] : 0.0;
            re += a * cos(t) - b * sin(t); im += a * sin(t) + b * cos(t);
        }
        out[2 * k] = inv ? re / N : re; out[2 * k + 1] = inv ? im / N : im;
    }
    return out;
}
static void run(const variant & v, const mtmd_audio_cache & cache, const fcase & c, float * in, float * out) {
    (c.inv ? v.ifft : v.fft)(cache, in, c.N, out);
}
static std::vector<float> load_wav(const char * p) {
    std::vector<float> w; FILE * f = fopen(p, "rb"); if (!f) return w;
    fseek(f, 44, SEEK_SET); int16_t s; while (fread(&s, 2, 1, f) == 1) w.push_back(s / 32768.0f); fclose(f); return w;
}
int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "parity";
    if (mode == "parity") {
        std::vector<float> wav = argc > 2 ? load_wav(argv[2]) : std::vector<float>();
        const variant & ref = find("old");
        printf("%-30s %-6s %5s %-8s %6s %10s %10s %8s %9s %9s %9s\n", "case", "dir", "N", "variant", "exact", "max_abs", "max_rel", "SNR_old", "errO_dB", "errN_dB", "outs_ok");
        for (const auto & c : g_cases) {
            mtmd_audio_cache cache; cache.fill_sin_cos_table(c.N);
            auto ins = make_inputs(c, wav, 20, 1234 + c.N);
            std::vector<std::vector<double>> refd; for (auto & x : ins) refd.push_back(dft_ref(x, c.N, c.inv));
            for (int vi = 0; vi < g_n_variants; vi++) {
                const variant & v = g_variants[vi];
                if (std::string(v.name) == "old") continue;
                int exact = 0; double max_abs = 0, max_rel = 0, sig = 0, noi = 0, eo = 0, en = 0, sr = 0;
                bool det = true;
                for (size_t t = 0; t < ins.size(); t++) {
                    std::vector<float> ia(8 * c.N, 0.f), ib(8 * c.N, 0.f), oa(8 * c.N, 0.f), ob(8 * c.N, 0.f), ob2(8 * c.N, 0.f);
                    for (size_t i = 0; i < ins[t].size(); i++) ia[i] = ib[i] = (float) ins[t][i];
                    std::vector<float> ib0 = ib;
                    run(ref, cache, c, ia.data(), oa.data());
                    run(v, cache, c, ib.data(), ob.data());
                    std::vector<float> ib1 = ib0; run(v, cache, c, ib1.data(), ob2.data());
                    if (memcmp(ob.data(), ob2.data(), 8 * c.N) != 0) det = false;
                    if (memcmp(oa.data(), ob.data(), sizeof(float) * 2 * c.N) == 0) exact++;
                    double mx = 0, frame_max = 0;
                    for (int i = 0; i < 2 * c.N; i++) {
                        double d = (double) ob[i] - oa[i]; mx = std::max(mx, fabs(d)); frame_max = std::max(frame_max, fabs((double) oa[i]));
                        sig += (double) oa[i] * oa[i]; noi += d * d;
                        double r = refd[t][i]; sr += r * r; eo += (oa[i] - r) * (oa[i] - r); en += (ob[i] - r) * (ob[i] - r);
                    }
                    max_abs = std::max(max_abs, mx); max_rel = std::max(max_rel, mx / frame_max);
                }
                auto db = [](double s, double n) { return n == 0 ? 999.0 : 10 * log10(s / n); };
                printf("%-30s %-6s %5d %-8s %3d/%-3zu %10.3g %10.3g %8.1f %9.1f %9.1f %9s\n", c.who, c.inv ? "ifft" : "fft", c.N, v.name, exact, ins.size(),
                       max_abs, max_rel, db(sig, noi), db(sr, eo), db(sr, en), det ? "det" : "NONDET");
            }
        }
        printf("SNR_old: new vs old output; errO_dB / errN_dB: SNR of old / new output against a double-precision DFT (higher is better)\n");
    } else if (mode == "sweep") {
        // every N in 1..300 plus some large ones, sin/cos table of N and 2 N points, both directions, random input
        const variant & ref = find("old");
        std::vector<int> sizes; for (int n = 1; n <= 300; n++) sizes.push_back(n);
        for (int n : { 384, 400, 480, 512, 640, 736, 768, 960, 1000, 1024, 1280, 1920, 2048, 2208, 3000, 4096 }) sizes.push_back(n);
        for (int vi = 0; vi < g_n_variants; vi++) {
            const variant & v = g_variants[vi];
            if (std::string(v.name) == "old") continue;
            int n_cases = 0, n_exact = 0; double worst_rel = 0, worst_db = 999; int worst_n = 0;
            for (int N : sizes) for (int tf : { 1, 2 }) for (int inv = 0; inv < 2; inv++) {
                mtmd_audio_cache cache; cache.fill_sin_cos_table(N * tf);
                std::mt19937 rng(N * 10 + tf + inv); std::uniform_real_distribution<float> u(-1, 1);
                std::vector<float> ia(8 * N, 0.f), oa(8 * N, 0.f), ob(8 * N, 0.f);
                for (int i = 0; i < (inv ? 2 * N : N); i++) ia[i] = u(rng);
                std::vector<float> ib = ia;
                (inv ? ref.ifft : ref.fft)(cache, ia.data(), N, oa.data());
                (inv ? v.ifft : v.fft)(cache, ib.data(), N, ob.data());
                double mx = 0, fm = 0, sg = 0, ns = 0;
                for (int i = 0; i < 2 * N; i++) { double d = (double) ob[i] - oa[i]; mx = std::max(mx, fabs(d)); fm = std::max(fm, fabs((double) oa[i])); sg += (double) oa[i] * oa[i]; ns += d * d; }
                n_cases++; if (memcmp(oa.data(), ob.data(), sizeof(float) * 2 * N) == 0) n_exact++;
                if (mx / fm > worst_rel) { worst_rel = mx / fm; worst_n = N; }
                if (ns > 0) worst_db = std::min(worst_db, 10 * log10(sg / ns));
            }
            printf("sweep %-8s cases=%d bit-exact=%d worst max_rel=%.3g (N=%d) worst SNR vs old=%.1f dB\n", v.name, n_cases, n_exact, worst_rel, worst_n, worst_db);
        }
    } else if (mode == "bench") {
        const variant & a = find(argv[2]); const variant & b = find(argv[3]);
        const int iters = argc > 4 ? atoi(argv[4]) : 41;
        printf("%-30s %-5s %5s | %10s %10s %10s | %10s %10s %10s | %7s %7s %7s %7s\n", "case", "dir", "N", "A_med_ns", "A_q25", "A_q75", "B_med_ns", "B_q25", "B_q75", "ratio", "pr_q25", "pr_med", "pr_q75");
        for (const auto & c : g_cases) {
            mtmd_audio_cache cache; cache.fill_sin_cos_table(c.N);
            if (c.N < 16) continue;
            auto ins = make_inputs(c, std::vector<float>(), 1, 99);
            std::vector<float> in(8 * c.N, 0.f), out(8 * c.N, 0.f);
            for (size_t i = 0; i < ins[0].size(); i++) in[i] = (float) ins[0][i];
            // calls per sample: about 2 ms of work
            double t0 = now_ns(); int k0 = 0; while (now_ns() - t0 < 2e6) { run(a, cache, c, in.data(), out.data()); k0++; }
            const int K = std::max(5, k0);
            auto sample = [&](const variant & v) { double s = now_ns(); for (int k = 0; k < K; k++) run(v, cache, c, in.data(), out.data()); return (now_ns() - s) / K; };
            sample(a); sample(b);
            std::vector<double> ta, tb, r;
            for (int i = 0; i < iters; i++) {
                double x, y;
                if (i % 2 == 0) { x = sample(a); y = sample(b); } else { y = sample(b); x = sample(a); }
                ta.push_back(x); tb.push_back(y); r.push_back(y / x);
            }
            printf("%-30s %-5s %5d | %10.0f %10.0f %10.0f | %10.0f %10.0f %10.0f | %7.3f %7.3f %7.3f %7.3f\n", c.who, c.inv ? "ifft" : "fft", c.N,
                   pct(ta, .5), pct(ta, .25), pct(ta, .75), pct(tb, .5), pct(tb, .25), pct(tb, .75), pct(tb, .5) / pct(ta, .5), pct(r, .25), pct(r, .5), pct(r, .75));
        }
        printf("A=%s B=%s iters=%d, ns per call, ratio = B/A medians, pr = per-pair ratio quartiles\n", a.name, b.name, iters);
    }
    return 0;
}
