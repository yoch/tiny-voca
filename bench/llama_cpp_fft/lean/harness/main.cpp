// A/B harness for tiny-voca host synth candidates (each compiled from its own tiny-voca.cpp, see wrap.sh)
//   main parity  dumps...               : every candidate vs ref on every dump + 1..3-frame cuts
//   main ab A B iters dumps...          : interleaved pairs (ABBA), whole workload per sample, ratio stats
#include "shim.h"
#include <chrono>
#include <map>

std::map<std::string, synth_fn> & registry();

struct utt { std::string name; std::vector<float> voc, f0; };

static std::vector<utt> load(int argc, char ** argv, int first) {
    std::vector<utt> u;
    for (int a = first; a < argc; a++) {
        FILE * fp = fopen(argv[a], "rb");
        int64_t nf = 0;
        if (!fp || fread(&nf, 8, 1, fp) != 1) { fprintf(stderr, "bad %s\n", argv[a]); exit(1); }
        utt x; x.name = strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a];
        x.f0.resize(nf); x.voc.resize(nf * tiny_voca_consts::n_voc);
        if (fread(x.f0.data(), 4, nf, fp) != (size_t) nf || fread(x.voc.data(), 4, x.voc.size(), fp) != x.voc.size()) exit(1);
        fclose(fp);
        u.push_back(x);
    }
    return u;
}

static std::vector<float> run(synth_fn f, const utt & u) {
    std::mt19937 rng(7);
    std::vector<float> out;
    f(u.voc, u.f0, rng, out);
    return out;
}

static int q16(float x) { return (int) lrintf(std::clamp(x, -1.0f, 1.0f) * 32767.0f); }

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double run_all(synth_fn f, const std::vector<utt> & us) {
    const double t0 = now_ms();
    for (auto & u : us) { std::mt19937 rng(7); std::vector<float> out; f(u.voc, u.f0, rng, out); }
    return now_ms() - t0;
}

static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); double i = p * (v.size() - 1); size_t lo = (size_t) i; return lo + 1 < v.size() ? v[lo] + (i - lo) * (v[lo + 1] - v[lo]) : v[lo]; }

int main(int argc, char ** argv) {
    auto & reg = registry();
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "list") { for (auto & [n, f] : reg) printf("%s\n", n.c_str()); return 0; }
    if (mode == "parity") {
        auto us = load(argc, argv, 2);
        for (int nf : {1, 2, 3}) {
            utt c; c.name = "cut" + std::to_string(nf);
            c.voc.assign(us[0].voc.begin(), us[0].voc.begin() + nf * tiny_voca_consts::n_voc);
            c.f0.assign(us[0].f0.begin(), us[0].f0.begin() + nf);
            us.push_back(c);
        }
        printf("%-10s %-8s %6s %9s %8s %8s %6s\n", "cand", "utt", "frames", "same_len", "snr_db", "max_lsb", "determ");
        for (auto & [n, f] : reg) {
            if (n == "ref") continue;
            double worst_snr = 1e9; int worst_lsb = 0; bool all_ok = true;
            for (auto & u : us) {
                auto r = run(reg["ref"], u), c = run(f, u), c2 = run(f, u);
                bool same = r.size() == c.size(), det = c == c2;
                double s = 0, e = 0; int m = 0;
                for (size_t i = 0; i < std::min(r.size(), c.size()); i++) { s += (double) r[i] * r[i]; e += ((double) r[i] - c[i]) * ((double) r[i] - c[i]); m = std::max(m, std::abs(q16(r[i]) - q16(c[i]))); }
                double snr = e == 0 ? INFINITY : 10 * log10(s / e);
                printf("%-10s %-8s %6zu %9s %8.1f %8d %6s\n", n.c_str(), u.name.c_str(), u.f0.size(), same ? "yes" : "NO", snr, m, det ? "yes" : "NO");
                worst_snr = std::min(worst_snr, snr); worst_lsb = std::max(worst_lsb, m);
                all_ok = all_ok && same && det && snr >= 110 && m <= 1;
            }
            printf("%-10s SUMMARY worst_snr=%.1f dB worst_lsb=%d %s\n", n.c_str(), worst_snr, worst_lsb, all_ok ? "PASS" : "FAIL");
        }
        return 0;
    }
    if (mode == "ab") {
        synth_fn fa = reg.at(argv[2]), fb = reg.at(argv[3]);
        int iters = atoi(argv[4]);
        auto us = load(argc, argv, 5);
        run_all(fa, us); run_all(fb, us); // warm up
        std::vector<double> ta, tb, r;
        for (int i = 0; i < iters; i++) {
            double a, b;
            if (i % 2 == 0) { a = run_all(fa, us); b = run_all(fb, us); } else { b = run_all(fb, us); a = run_all(fa, us); }
            ta.push_back(a); tb.push_back(b); r.push_back(b / a);
        }
        printf("ab %s %s iters=%d | %s med=%.3f ms | %s med=%.3f ms | ratio_of_medians=%.4f | pair_ratio q25=%.4f med=%.4f q75=%.4f\n",
               argv[2], argv[3], iters, argv[2], pct(ta, 0.5), argv[3], pct(tb, 0.5), pct(tb, 0.5) / pct(ta, 0.5), pct(r, 0.25), pct(r, 0.5), pct(r, 0.75));
        return 0;
    }
    fprintf(stderr, "usage: main list|parity|ab ...\n");
    return 1;
}
