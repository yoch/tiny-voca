// function-level timing and outputs of the other callers of the mtmd FFT, on the real libmtmd.so of a build
// (link against one build, run with LD_LIBRARY_PATH=<bins/build>, see other.sh)
//   other time WAV OUT.bin iters   ->  per function: median / q25 / q75 ms per call, outputs written to OUT.bin
//   other cmp A.bin B.bin          ->  max abs diff and SNR per output block
#include "mtmd-audio.h"
#include "mtmd-model.h"
#include "mtmd-image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); double i = p * (v.size() - 1); size_t lo = (size_t) i; return lo + 1 < v.size() ? v[lo] + (i - lo) * (v[lo + 1] - v[lo]) : v[lo]; }
static std::vector<float> load_wav(const char * p) {
    std::vector<float> w; FILE * f = fopen(p, "rb"); if (!f) { fprintf(stderr, "no wav\n"); exit(1); }
    fseek(f, 44, SEEK_SET); int16_t s; while (fread(&s, 2, 1, f) == 1) w.push_back(s / 32768.0f); fclose(f); return w;
}
// half spectra (n_fft / 2 + 1 complex) of hann windowed frames of x, hop
static std::vector<float> stft(const std::vector<float> & x, int n_fft, int hop, int n_frames) {
    const int nb = n_fft / 2 + 1;
    std::vector<float> s((size_t) n_frames * nb * 2);
    std::vector<double> c(n_fft), sn(n_fft);
    for (int i = 0; i < n_fft; i++) { c[i] = cos(2 * M_PI * i / n_fft); sn[i] = sin(2 * M_PI * i / n_fft); }
    for (int f = 0; f < n_frames; f++) {
        for (int k = 0; k < nb; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < n_fft; n++) {
                size_t idx = ((size_t) f * hop + n) % x.size();
                double v = x[idx] * (0.5 - 0.5 * cos(2 * M_PI * n / n_fft));
                int q = (int) ((int64_t) k * n % n_fft);
                re += v * c[q]; im -= v * sn[q];
            }
            s[((size_t) f * nb + k) * 2] = (float) re; s[((size_t) f * nb + k) * 2 + 1] = (float) im;
        }
    }
    return s;
}
struct job { std::string name; std::function<std::vector<float>()> run; };

int main(int argc, char ** argv) {
    std::string mode = argv[1];
    if (mode == "cmp") {
        FILE * fa = fopen(argv[2], "rb"); FILE * fb = fopen(argv[3], "rb");
        char na[64], nb[64]; uint64_t la, lb;
        printf("%-28s %10s %12s %12s %10s %8s\n", "output", "values", "max_abs", "max|ref|", "SNR_dB", "exact");
        while (fread(na, 64, 1, fa) == 1 && fread(nb, 64, 1, fb) == 1) {
            if (fread(&la, 8, 1, fa) != 1 || fread(&lb, 8, 1, fb) != 1) return 1;
            std::vector<float> a(la), b(lb);
            if (fread(a.data(), 4, la, fa) != la || fread(b.data(), 4, lb, fb) != lb) return 1;
            if (la != lb) { printf("%-28s length %zu vs %zu\n", na, (size_t) la, (size_t) lb); continue; }
            double mx = 0, mr = 0, sg = 0, ns = 0;
            for (size_t i = 0; i < la; i++) { double d = (double) b[i] - a[i]; mx = std::max(mx, fabs(d)); mr = std::max(mr, fabs((double) a[i])); sg += (double) a[i] * a[i]; ns += d * d; }
            printf("%-28s %10zu %12.3g %12.3g %10.1f %8s\n", na, (size_t) la, mx, mr, ns == 0 ? 999.0 : 10 * log10(sg / ns), memcmp(a.data(), b.data(), 4 * la) == 0 ? "yes" : "no");
        }
        return 0;
    }
    std::vector<float> wav = load_wav(argv[2]);
    const int iters = argc > 4 ? atoi(argv[4]) : 15;
    std::vector<job> jobs;

    // chatterbox speaker reference: kaldi fbank (512), s3 tokenizer log-mel x2 (400), matcha log-mel (1920), voice encoder mel (400)
    static mtmd_model model;
    model.hparams.n_mel_bins = 80;
    model.hparams.audio_sample_rate = 16000;
    static mtmd_audio_preprocessor_chatterbox_ref cb(&model, true);
    cb.initialize();
    std::vector<float> pcm16;   // 10 s at 16 kHz: the speech of the wav, looped
    for (size_t i = 0; i < 160000; i++) pcm16.push_back(wav[i % wav.size()]);
    jobs.push_back({ "chatterbox_ref preprocess 10s", [&]() {
        auto out = cb.preprocess(pcm16.data(), pcm16.size());
        std::vector<float> r;
        for (auto & e : out.entries) { auto & buf = e.get_ro_buf(); r.insert(r.end(), buf.begin(), buf.end()); }
        return r;
    } });

    // voxtral realtime: mtmd_audio_log_mel_frame, n_fft 400, 128 mels, 500 frames (5 s at hop 160)
    static mtmd_audio_cache lm;
    lm.fill_sin_cos_table(400); lm.fill_hann_window(400, true); lm.fill_mel_filterbank_matrix(128, 400, 16000);
    jobs.push_back({ "log_mel_frame 400 x500", [&]() {
        std::vector<float> r(500 * 128);
        for (int f = 0; f < 500; f++) mtmd_audio_log_mel_frame(lm, pcm16.data() + f * 160, 400, 128, r.data() + f * 128);
        return r;
    } });

    // streaming istft: lfm3a (1280 / 320) and soprano-like (2048 / 512), 200 frames of real-audio spectra
    for (auto cfg : { std::pair<int, int>{ 1280, 320 }, std::pair<int, int>{ 2048, 512 } }) {
        const int n_fft = cfg.first, hop = cfg.second, nf = 200;
        auto spec = std::make_shared<std::vector<float>>(stft(wav, n_fft, hop, nf));
        jobs.push_back({ "istft " + std::to_string(n_fft) + " x200", [=]() {
            mtmd_audio_streaming_istft st(n_fft, hop);
            std::vector<float> r;
            for (int f = 0; f < nf; f++) { auto o = st.process_frame(spec->data() + (size_t) f * (n_fft / 2 + 1) * 2); r.insert(r.end(), o.begin(), o.end()); }
            auto o = st.flush(); r.insert(r.end(), o.begin(), o.end());
            return r;
        } });
    }

    // zonos3 synth: 2 branches 736 / 2208, hop 184; synthetic mixer / resampler kernels, log-mag and phase from real audio
    {
        const int nf = 120, n0 = 736, n1 = 2208, hop = 184;
        std::mt19937 rng(3); std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        auto mixer = std::make_shared<std::vector<float>>(2 * 31); for (auto & v : *mixer) v = u(rng);
        auto rsk = std::make_shared<std::vector<float>>((size_t) 735 * (736 + 2 * 8)); for (auto & v : *rsk) v = u(rng) * 0.01f;
        auto s0 = stft(wav, n0, hop, nf), s1 = stft(wav, n1, hop, nf);
        const int rows = (n0 + 2) + (n1 + 2);
        auto spec = std::make_shared<std::vector<float>>((size_t) nf * rows);
        for (int c = 0; c < nf; c++) {
            float * col = spec->data() + (size_t) c * rows;
            for (int b = 0; b < 2; b++) {
                const int n = b ? n1 : n0, bins = n / 2 + 1; const auto & s = b ? s1 : s0;
                float * lm_ = col + (b ? n0 + 2 : 0), * ph = lm_ + bins;
                for (int k = 0; k < bins; k++) { float re = s[((size_t) c * bins + k) * 2], im = s[((size_t) c * bins + k) * 2 + 1]; lm_[k] = 0.5f * logf(re * re + im * im + 1e-12f); ph[k] = atan2f(im, re); }
            }
        }
        jobs.push_back({ "zonos3 synth x120", [=]() {
            mtmd_audio_zonos3_synth z(*mixer, *rsk, n0, n1, hop, 736, 735);
            std::vector<float> r, o;
            z.push(spec->data(), nf, o); r.insert(r.end(), o.begin(), o.end());
            o.clear(); z.flush(o); r.insert(r.end(), o.begin(), o.end());
            return r;
        } });
    }

    FILE * fo = fopen(argv[3], "wb");
    for (auto & j : jobs) {
        std::vector<float> r = j.run();
        std::vector<double> t;
        for (int i = 0; i < iters; i++) { double t0 = now_ms(); j.run(); t.push_back(now_ms() - t0); }
        printf("%-30s med=%9.3f ms q25=%9.3f q75=%9.3f values=%zu\n", j.name.c_str(), pct(t, .5), pct(t, .25), pct(t, .75), r.size());
        char nm[64] = {0}; snprintf(nm, 64, "%s", j.name.c_str()); uint64_t n = r.size();
        fwrite(nm, 64, 1, fo); fwrite(&n, 8, 1, fo); fwrite(r.data(), 4, n, fo);
    }
    fclose(fo);
    return 0;
}
