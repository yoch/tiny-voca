// Local benchmark: llama.cpp tiny-voca host synth (reference copy) vs two faster variants.
// ref : verbatim copy of tools/mtmd/models/tiny-voca.cpp synth + mtmd-audio.cpp FFT (llama.cpp 6aee508)
// A   : same FFT, tables built once, filter response computed once per envelope (noise: once per frame)
// B   : A + packed real FFT (N/2 complex, iterative, bit-reversed, precomputed twiddles), from E04 vocoder_dsp.c
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <cstring>
#include <thread>

static constexpr int   hop = 256, n_cep_causal = 80, n_cep_anti = 16, n_noise_cep = 32;
static constexpr int   harm_fft = 512, harm_hop = 128, harm_steps = hop / harm_hop;
static constexpr int   noise_fft = 256, noise_hop = 64, noise_steps = hop / noise_hop;
static constexpr int   n_voc = n_cep_causal + n_cep_anti + harm_steps + n_noise_cep + noise_steps;
static constexpr float sample_rate = 24000.0f, harmonic_limit = 0.47f;

// ------------------------------------------------------------------ reference (copied)
namespace ref {
struct cache_t {
    std::vector<float> sin_vals, cos_vals, hann_window;
    void fill_sin_cos_table(uint32_t n) {
        sin_vals.resize(n); cos_vals.resize(n);
        for (uint32_t i = 0; i < n; i++) { double theta = (2 * M_PI * i) / n; sin_vals[i] = sinf(theta); cos_vals[i] = cosf(theta); }
    }
    void fill_hann_window(uint32_t length, bool periodic) {
        hann_window.resize(length); int offset = periodic ? 0 : -1;
        for (uint32_t i = 0; i < length; i++) hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
};

template <bool Inverse, bool RealInput>
static void fft_impl(const cache_t & cache, float * in, int N, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    if (N == 1) { out[0] = in[0]; if constexpr (RealInput) out[1] = 0.0f; else out[1] = in[1]; return; }
    const int half_N = N / 2;
    if constexpr (RealInput) {
        float * even = in + N;
        for (int i = 0; i < half_N; ++i) even[i] = in[2 * i];
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, true>(cache, even, half_N, even_fft);
        float * odd = even;
        for (int i = 0; i < half_N; ++i) odd[i] = in[2 * i + 1];
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, true>(cache, odd, half_N, odd_fft);
    } else {
        float * even = in + N * 2;
        for (int i = 0; i < half_N; ++i) { even[i * 2 + 0] = in[2 * i * 2 + 0]; even[i * 2 + 1] = in[2 * i * 2 + 1]; }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, false>(cache, even, half_N, even_fft);
        float * odd = even;
        for (int i = 0; i < half_N; ++i) { odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0]; odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1]; }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, false>(cache, odd, half_N, odd_fft);
    }
    float * even_fft = out + 2 * N;
    float * odd_fft  = even_fft + N;
    const int sin_cos_step = n_sin_cos_vals / N;
    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step;
        float re = cache.cos_vals[idx], im = sign * cache.sin_vals[idx];
        float re_odd = odd_fft[2 * k + 0], im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
        out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);
        out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
        out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
    }
}
static void fft(const cache_t & c, float * in, int N, float * out) { fft_impl<false, true>(c, in, N, out); }
static void ifft(const cache_t & c, float * in, int N, float * out) { fft_impl<true, false>(c, in, N, out); }

void pulse_train(const std::vector<float> & f0, std::vector<float> & out) {
    const int64_t n_frames = (int64_t) f0.size();
    std::vector<double> knots(n_frames + 1);
    for (int64_t i = 0; i < n_frames; i++) knots[i] = std::log(std::max((double) f0[i], 1.0));
    knots[n_frames] = knots[n_frames - 1];
    out.resize((size_t) (n_frames * hop));
    double acc = 0.0;
    for (int64_t n = 0; n < n_frames * hop; n++) {
        const int64_t low = n / hop;
        const double  w   = (double) (n - low * hop) / hop;
        const double  freq = std::exp(knots[low] * (1.0 - w) + knots[low + 1] * w);
        acc += freq / sample_rate;
        const double theta = std::fmod(2.0 * M_PI * acc, 2.0 * M_PI);
        const double harmonics = std::max((double) harmonic_limit * sample_rate / freq, 1.0);
        const double whole = std::floor(harmonics);
        const double denom = 2.0 * std::sin(theta / 2.0);
        const double core  = std::fabs(denom) < 1e-7 ? whole : std::sin((whole + 0.5) * theta) / denom - 0.5;
        const double wave  = core + (harmonics - whole) * std::cos((whole + 1.0) * theta);
        out[n] = (float) (wave * std::sqrt(2.0 / harmonics));
    }
}

static void filter(const std::vector<float> & src, int n_fft, int hop, const std::function<void(int64_t, float *)> & cep, std::vector<float> & out) {
    const int     n_bins = n_fft / 2 + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    cache_t cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    const float * win = cache.hann_window.data();
    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);
    std::vector<float> ola((size_t) n_full, 0.0f), env((size_t) n_full, 0.0f);
    std::vector<float> fin(8 * n_fft), fout(8 * n_fft), rin(8 * n_fft), rout(8 * n_fft), cbuf(n_fft);
    std::vector<float> resp(2 * n_bins);
    for (int64_t j = 0; j < n_stft; j++) {
        std::fill(cbuf.begin(), cbuf.end(), 0.0f);
        cep(j, cbuf.data());
        std::copy(cbuf.begin(), cbuf.end(), rin.begin());
        fft(cache, rin.data(), n_fft, rout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(rout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(rout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(rout[2 * k + 1]);
        }
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) fin[i] = seg[i] * win[i];
        fft(cache, fin.data(), n_fft, fout.data());
        for (int k = 0; k < n_bins; k++) {
            const float a = fout[2 * k], b = fout[2 * k + 1];
            const float c = resp[2 * k], d = resp[2 * k + 1];
            rin[2 * k + 0] = a * c - b * d;
            rin[2 * k + 1] = (k == 0 || k == n_fft / 2) ? 0.0f : a * d + b * c;
        }
        for (int k = 1; k < n_fft - n_bins + 1; k++) { rin[2 * (n_fft - k) + 0] = rin[2 * k + 0]; rin[2 * (n_fft - k) + 1] = -rin[2 * k + 1]; }
        ifft(cache, rin.data(), n_fft, rout.data());
        float * o = ola.data() + j * hop;
        float * e = env.data() + j * hop;
        for (int i = 0; i < n_fft; i++) { o[i] += rout[2 * i] * win[i]; e[i] += win[i] * win[i]; }
    }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) { const int64_t k = i + n_fft / 2; out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f; }
}

static float lifter(int i, int first) { return i == 0 && first == 0 ? 2.0f : 1.0f / (1.0f + (float) (i + first) / 4.0f); }

void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const int split = n_cep_causal + n_cep_anti;
    auto harm_cep = [&](int64_t j, float * buf) {
        const int64_t s    = std::min(j, harm_steps * n_frames - 1);
        const float * here = voc.data() + (s / harm_steps) * n_voc;
        const float * next = voc.data() + std::min((s + 1) / harm_steps, n_frames - 1) * n_voc;
        for (int i = 0; i < n_cep_causal; i++) buf[i] = 0.5f * (here[i] + next[i]) * lifter(i, 0);
        buf[0] += here[split + s % harm_steps];
        for (int i = 0; i < n_cep_anti; i++) buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * lifter(i, 1);
    };
    auto noise_cep = [&](int64_t j, float * buf) {
        const int64_t s   = std::min(j, noise_steps * n_frames - 1);
        const float * row = voc.data() + (s / noise_steps) * n_voc + split + harm_steps;
        for (int i = 0; i < n_noise_cep; i++) buf[i] = row[i] * lifter(i, 0);
        buf[0] += row[n_noise_cep + s % noise_steps];
    };
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : noise) v = dist(rng);
    filter(pulses, harm_fft, harm_hop, harm_cep, harm_out);
    filter(noise, noise_fft, noise_hop, noise_cep, noise_out);
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace ref

// ------------------------------------------------------------------ shared: envelope cepstrum without the gain
// harmonic envelope of sub-frame s (interpolated between frames), noise envelope of frame t; the gain is added to the log magnitude later
static void harm_envelope(const float * voc, int64_t n_frames, int64_t s, float * buf) {
    const float * here = voc + (s / harm_steps) * n_voc;
    const float * next = voc + std::min((s + 1) / harm_steps, n_frames - 1) * n_voc;
    for (int i = 0; i < n_cep_causal; i++) buf[i] = 0.5f * (here[i] + next[i]) * ref::lifter(i, 0);
    for (int i = 0; i < n_cep_anti; i++) buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * ref::lifter(i, 1);
}
static void noise_envelope(const float * voc, int64_t t, float * buf) {
    const float * row = voc + t * n_voc + n_cep_causal + n_cep_anti + harm_steps;
    for (int i = 0; i < n_noise_cep; i++) buf[i] = row[i] * ref::lifter(i, 0);
}
static float harm_gain(const float * voc, int64_t s)  { return voc[(s / harm_steps) * n_voc + n_cep_causal + n_cep_anti + s % harm_steps]; }
static float noise_gain(const float * voc, int64_t s) { return voc[(s / noise_steps) * n_voc + n_cep_causal + n_cep_anti + harm_steps + n_noise_cep + s % noise_steps]; }

// ------------------------------------------------------------------ A: restructured, same FFT
namespace A {
struct plan_t { ref::cache_t cache; int n; };
static const plan_t & plan(int n) {
    static plan_t p512 = [] { plan_t p; p.n = 512; p.cache.fill_sin_cos_table(512); p.cache.fill_hann_window(512, true); return p; }();
    static plan_t p256 = [] { plan_t p; p.n = 256; p.cache.fill_sin_cos_table(256); p.cache.fill_hann_window(256, true); return p; }();
    return n == 512 ? p512 : p256;
}

// env(key, buf) writes the cepstrum of envelope `key` without the gain; key(j) and gain(j) give them for STFT frame j
template <typename Key, typename Env, typename Gain>
static void filter(const std::vector<float> & src, int n_fft, int hop, Key key, Env env_cep, Gain gain, std::vector<float> & out) {
    const plan_t & p = plan(n_fft);
    const int     n_bins = n_fft / 2 + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    const float * win = p.cache.hann_window.data();
    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);
    std::vector<float> ola((size_t) n_full, 0.0f);
    std::vector<float> fin(8 * n_fft), fout(8 * n_fft), rin(8 * n_fft), rout(8 * n_fft), cbuf(n_fft);
    std::vector<float> logmag(n_bins), pc(n_bins), ps(n_bins), resp(2 * n_bins);
    int64_t last_key = -1;
    for (int64_t j = 0; j < n_stft; j++) {
        const int64_t k_env = key(j);
        if (k_env != last_key) {
            std::fill(cbuf.begin(), cbuf.end(), 0.0f);
            env_cep(k_env, cbuf.data());
            std::copy(cbuf.begin(), cbuf.end(), rin.begin());
            ref::fft(p.cache, rin.data(), n_fft, rout.data());
            for (int k = 0; k < n_bins; k++) { logmag[k] = rout[2 * k]; pc[k] = std::cos(rout[2 * k + 1]); ps[k] = std::sin(rout[2 * k + 1]); }
            last_key = k_env;
        }
        const float g = gain(j);
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(logmag[k] + g, -14.0f, 6.0f));
            resp[2 * k + 0] = mag * pc[k];
            resp[2 * k + 1] = mag * ps[k];
        }
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) fin[i] = seg[i] * win[i];
        ref::fft(p.cache, fin.data(), n_fft, fout.data());
        for (int k = 0; k < n_bins; k++) {
            const float a = fout[2 * k], b = fout[2 * k + 1], c = resp[2 * k], d = resp[2 * k + 1];
            rin[2 * k + 0] = a * c - b * d;
            rin[2 * k + 1] = (k == 0 || k == n_fft / 2) ? 0.0f : a * d + b * c;
        }
        for (int k = 1; k < n_fft - n_bins + 1; k++) { rin[2 * (n_fft - k) + 0] = rin[2 * k + 0]; rin[2 * (n_fft - k) + 1] = -rin[2 * k + 1]; }
        ref::ifft(p.cache, rin.data(), n_fft, rout.data());
        float * o = ola.data() + j * hop;
        for (int i = 0; i < n_fft; i++) o[i] += rout[2 * i] * win[i];
    }
    // window power sum: 4x overlap of a periodic Hann; computed as the reference does, once
    std::vector<float> env((size_t) n_full, 0.0f);
    for (int64_t j = 0; j < n_stft; j++) { float * e = env.data() + j * hop; for (int i = 0; i < n_fft; i++) e[i] += win[i] * win[i]; }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) { const int64_t k = i + n_fft / 2; out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f; }
}

void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    ref::pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
           [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace A

// ------------------------------------------------------------------ B: A + packed real FFT (from E04 vocoder_dsp.c)
namespace B {
struct plan_t {
    int n, m;
    std::vector<int> reverse;
    std::vector<float> cosines, sines, rot_r, rot_i, window;
    explicit plan_t(int n_) : n(n_), m(n_ / 2), reverse(m), cosines(m), sines(m), rot_r(m + 1), rot_i(m + 1), window(n_) {
        for (int k = 0; k < m; ++k) { int r = 0; for (int b = m, i = k; b > 1; b >>= 1, i >>= 1) r = 2 * r + (i & 1); reverse[k] = r; }
        for (int half = 1; half < m; half *= 2)
            for (int j = 0; j < half; ++j) { cosines[half - 1 + j] = cosf(-2.0f * (float) M_PI * j / (2 * half)); sines[half - 1 + j] = sinf(-2.0f * (float) M_PI * j / (2 * half)); }
        for (int k = 0; k <= m; ++k) { rot_r[k] = cosf(-2.0f * (float) M_PI * k / n); rot_i[k] = sinf(-2.0f * (float) M_PI * k / n); }
        for (int i = 0; i < n; i++) window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / n));
    }
};
static const plan_t & plan(int n) { static plan_t p512(512), p256(256); return n == 512 ? p512 : p256; }

static void butterflies(const plan_t & p, float * __restrict r, float * __restrict i) {
    for (int half = 1; half < p.m; half *= 2) {
        const float * c = p.cosines.data() + half - 1;
        const float * s = p.sines.data() + half - 1;
        for (int block = 0; block < p.m; block += 2 * half)
            for (int j = 0; j < half; ++j) {
                const int a = block + j, b = a + half;
                const float tr = r[b] * c[j] - i[b] * s[j], ti = r[b] * s[j] + i[b] * c[j];
                r[b] = r[a] - tr; i[b] = i[a] - ti; r[a] += tr; i[a] += ti;
            }
    }
}
// rfft of n real samples (x[0..n)), out: m+1 bins
static void rfft(const plan_t & p, const float * x, float * outr, float * outi) {
    float r[256], i[256];
    for (int j = 0; j < p.m; ++j) { const int a = 2 * p.reverse[j]; r[j] = x[a]; i[j] = x[a + 1]; }
    butterflies(p, r, i);
    outr[0] = r[0] + i[0]; outr[p.m] = r[0] - i[0]; outi[0] = outi[p.m] = 0;
    for (int k = 1; k < p.m; ++k) {
        const float dr = r[k] - r[p.m - k], di = i[k] + i[p.m - k];
        outr[k] = 0.5f * (r[k] + r[p.m - k] + di * p.rot_r[k] + dr * p.rot_i[k]);
        outi[k] = 0.5f * (i[k] - i[p.m - k] - dr * p.rot_r[k] + di * p.rot_i[k]);
    }
}
// irfft of m+1 bins into n real samples (scaled 1/n)
static void irfft(const plan_t & p, const float * xr, const float * xi, float * y) {
    float r[256], i[256];
    for (int j = 0; j < p.m; ++j) {
        const int k = p.reverse[j], o = p.m - k;
        const float dr = xr[k] - xr[o], di = xi[k] + xi[o];
        r[j] = 0.5f * (xr[k] + xr[o] - di * p.rot_r[k] + dr * p.rot_i[k]);
        i[j] = -0.5f * (xi[k] - xi[o] + dr * p.rot_r[k] + di * p.rot_i[k]);
    }
    butterflies(p, r, i);
    const float scale = 1.0f / p.m;
    for (int j = 0; j < p.m; ++j) { y[2 * j] = r[j] * scale; y[2 * j + 1] = -i[j] * scale; }
}

template <typename Key, typename Env, typename Gain>
static void filter(const std::vector<float> & src, int n_fft, int hop, Key key, Env env_cep, Gain gain, std::vector<float> & out) {
    const plan_t & p = plan(n_fft);
    const int     n_bins = p.m + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    const float * win = p.window.data();
    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);
    std::vector<float> ola((size_t) n_full, 0.0f);
    float cbuf[512], fin[512], y[512], lr[257], li[257], pc[257], ps[257], sr[257], si[257];
    int64_t last_key = -1;
    for (int64_t j = 0; j < n_stft; j++) {
        const int64_t k_env = key(j);
        if (k_env != last_key) {
            std::fill(cbuf, cbuf + n_fft, 0.0f);
            env_cep(k_env, cbuf);
            rfft(p, cbuf, lr, li);
            for (int k = 0; k < n_bins; k++) { pc[k] = std::cos(li[k]); ps[k] = std::sin(li[k]); }
            last_key = k_env;
        }
        const float g = gain(j);
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) fin[i] = seg[i] * win[i];
        rfft(p, fin, sr, si);
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(lr[k] + g, -14.0f, 6.0f));
            const float c = mag * pc[k], d = mag * ps[k], a = sr[k], b = si[k];
            sr[k] = a * c - b * d;
            si[k] = a * d + b * c;
        }
        si[0] = si[p.m] = 0.0f;
        irfft(p, sr, si, y);
        float * o = ola.data() + j * hop;
        for (int i = 0; i < n_fft; i++) o[i] += y[i] * win[i];
    }
    std::vector<float> env((size_t) n_full, 0.0f);
    for (int64_t j = 0; j < n_stft; j++) { float * e = env.data() + j * hop; for (int i = 0; i < n_fft; i++) e[i] += win[i] * win[i]; }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) { const int64_t k = i + n_fft / 2; out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f; }
}

void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    ref::pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
           [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace B


// ------------------------------------------------------------------ C: B + radix-4 first pass, precomputed overlap-add envelope, faster pulse train
namespace C {
using B::plan_t;
using B::plan;

static void butterflies(const plan_t & p, float * __restrict r, float * __restrict i) {
    // stages of size 2 and 4 together: twiddles 1 and -i only
    for (int a = 0; a < p.m; a += 4) {
        const float r0 = r[a] + r[a + 1], i0 = i[a] + i[a + 1], r1 = r[a] - r[a + 1], i1 = i[a] - i[a + 1];
        const float r2 = r[a + 2] + r[a + 3], i2 = i[a + 2] + i[a + 3], r3 = r[a + 2] - r[a + 3], i3 = i[a + 2] - i[a + 3];
        r[a] = r0 + r2; i[a] = i0 + i2; r[a + 2] = r0 - r2; i[a + 2] = i0 - i2;
        r[a + 1] = r1 + i3; i[a + 1] = i1 - r3; r[a + 3] = r1 - i3; i[a + 3] = i1 + r3;
    }
    for (int half = 4; half < p.m; half *= 2) {
        const float * c = p.cosines.data() + half - 1;
        const float * s = p.sines.data() + half - 1;
        for (int block = 0; block < p.m; block += 2 * half)
            for (int j = 0; j < half; ++j) {
                const int a = block + j, b = a + half;
                const float tr = r[b] * c[j] - i[b] * s[j], ti = r[b] * s[j] + i[b] * c[j];
                r[b] = r[a] - tr; i[b] = i[a] - ti; r[a] += tr; i[a] += ti;
            }
    }
}
static void rfft(const plan_t & p, const float * x, float * outr, float * outi) {
    float r[256], i[256];
    for (int j = 0; j < p.m; ++j) { const int a = 2 * p.reverse[j]; r[j] = x[a]; i[j] = x[a + 1]; }
    C::butterflies(p, r, i);
    outr[0] = r[0] + i[0]; outr[p.m] = r[0] - i[0]; outi[0] = outi[p.m] = 0;
    for (int k = 1; k < p.m; ++k) {
        const float dr = r[k] - r[p.m - k], di = i[k] + i[p.m - k];
        outr[k] = 0.5f * (r[k] + r[p.m - k] + di * p.rot_r[k] + dr * p.rot_i[k]);
        outi[k] = 0.5f * (i[k] - i[p.m - k] - dr * p.rot_r[k] + di * p.rot_i[k]);
    }
}
static void irfft(const plan_t & p, const float * xr, const float * xi, float * y) {
    float r[256], i[256];
    for (int j = 0; j < p.m; ++j) {
        const int k = p.reverse[j], o = p.m - k;
        const float dr = xr[k] - xr[o], di = xi[k] + xi[o];
        r[j] = 0.5f * (xr[k] + xr[o] - di * p.rot_r[k] + dr * p.rot_i[k]);
        i[j] = -0.5f * (xi[k] - xi[o] + dr * p.rot_r[k] + di * p.rot_i[k]);
    }
    C::butterflies(p, r, i);
    const float scale = 1.0f / p.m;
    for (int j = 0; j < p.m; ++j) { y[2 * j] = r[j] * scale; y[2 * j + 1] = -i[j] * scale; }
}

void pulse_train(const std::vector<float> & f0, std::vector<float> & out) {
    const int64_t n_frames = (int64_t) f0.size();
    std::vector<double> knots(n_frames + 1);
    for (int64_t i = 0; i < n_frames; i++) knots[i] = std::log(std::max((double) f0[i], 1.0));
    knots[n_frames] = knots[n_frames - 1];
    out.resize((size_t) (n_frames * hop));
    const double top = (double) harmonic_limit * sample_rate;
    double acc = 0.0;
    for (int64_t i = 0; i < n_frames; i++) {
        double freq = std::exp(knots[i]);
        const double ratio = std::exp((knots[i + 1] - knots[i]) / hop);
        for (int m = 0; m < hop; m++, freq *= ratio) {
            acc += freq / sample_rate;
            acc -= std::floor(acc);
            const double theta = 2.0 * M_PI * acc;
            const double harmonics = std::max(top / freq, 1.0);
            const double whole = std::floor(harmonics);
            const double h = theta / 2.0, a = (whole + 0.5) * theta;
            const double sh = std::sin(h), ch = std::cos(h), sa = std::sin(a), ca = std::cos(a);
            const double core = std::fabs(2.0 * sh) < 1e-7 ? whole : sa / (2.0 * sh) - 0.5;
            const double wave = core + (harmonics - whole) * (ca * ch - sa * sh);
            out[i * hop + m] = (float) (wave * std::sqrt(2.0 / harmonics));
        }
    }
}

template <typename Key, typename Env, typename Gain>
static void filter(const std::vector<float> & src, int n_fft, int hop, Key key, Env env_cep, Gain gain, std::vector<float> & out) {
    const plan_t & p = plan(n_fft);
    const int     n_bins = p.m + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    const float * win = p.window.data();
    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);
    std::vector<float> ola((size_t) n_full, 0.0f);
    float cbuf[512], fin[512], y[512], lr[257], li[257], pc[257], ps[257], sr[257], si[257];
    int64_t last_key = -1;
    for (int64_t j = 0; j < n_stft; j++) {
        const int64_t k_env = key(j);
        if (k_env != last_key) {
            std::fill(cbuf, cbuf + n_fft, 0.0f);
            env_cep(k_env, cbuf);
            C::rfft(p, cbuf, lr, li);
            for (int k = 0; k < n_bins; k++) { pc[k] = std::cos(li[k]); ps[k] = std::sin(li[k]); }
            last_key = k_env;
        }
        const float g = gain(j);
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) fin[i] = seg[i] * win[i];
        C::rfft(p, fin, sr, si);
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(lr[k] + g, -14.0f, 6.0f));
            const float c = mag * pc[k], d = mag * ps[k], a = sr[k], b = si[k];
            sr[k] = a * c - b * d;
            si[k] = a * d + b * c;
        }
        C::irfft(p, sr, si, y);
        float * o = ola.data() + j * hop;
        for (int i = 0; i < n_fft; i++) o[i] += y[i] * win[i];
    }
    // window power: three windows over the first kept hop, four after; same summation order as frame by frame
    float first[256], steady[256];
    for (int i = 0; i < hop; i++) {
        float e = 0.0f;
        for (int q = 2; q >= 0; q--) e += win[i + q * hop] * win[i + q * hop];
        first[i] = e;
        steady[i] = 0.0f + win[i + 3 * hop] * win[i + 3 * hop];
        steady[i] += win[i + 2 * hop] * win[i + 2 * hop];
        steady[i] += win[i + hop] * win[i + hop];
        steady[i] += win[i] * win[i];
    }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) {
        const int64_t k = i + n_fft / 2;
        out[i] = ola[k] / (i < hop ? first[i] : steady[i % hop]);
    }
}

void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
           [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace C

// ------------------------------------------------------------------ D: C + one cepstrum FFT and one sincos per harmonic frame (half-angle trick), no source copy
namespace D {
using B::plan_t;
using B::plan;

// response(j, lr, pc, ps) fills log magnitude (without gain) and phase cos/sin of STFT frame j
template <typename Response, typename Gain>
static void filter(const std::vector<float> & src, const plan_t & p, int hop, Response response, Gain gain, std::vector<float> & out) {
    const int     n_fft  = p.n;
    const int     n_bins = p.m + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    const float * win = p.window.data();
    std::vector<float> ola((size_t) n_full, 0.0f);
    float fin[512], y[512], lr[257], pc[257], ps[257], sr[257], si[257];
    for (int64_t j = 0; j < n_stft; j++) {
        response(j, lr, pc, ps);
        const float g = gain(j);
        // frame j covers padded samples [j * hop, j * hop + n_fft), the source starts at n_fft / 2
        const int64_t start = j * hop - n_fft / 2;
        if (start >= 0 && start + n_fft <= n_src) {
            for (int i = 0; i < n_fft; i++) fin[i] = src[start + i] * win[i];
        } else {
            for (int i = 0; i < n_fft; i++) { const int64_t t = start + i; fin[i] = t >= 0 && t < n_src ? src[t] * win[i] : 0.0f; }
        }
        C::rfft(p, fin, sr, si);
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(lr[k] + g, -14.0f, 6.0f));
            const float c = mag * pc[k], d = mag * ps[k], a = sr[k], b = si[k];
            sr[k] = a * c - b * d;
            si[k] = a * d + b * c;
        }
        C::irfft(p, sr, si, y);
        float * o = ola.data() + j * hop;
        for (int i = 0; i < n_fft; i++) o[i] += y[i] * win[i];
    }
    float first[256], steady[256];
    for (int i = 0; i < hop; i++) {
        float e = 0.0f;
        for (int q = 2; q >= 0; q--) e += win[i + q * hop] * win[i + q * hop];
        first[i] = e;
        steady[i] = 0.0f + win[i + 3 * hop] * win[i + 3 * hop];
        steady[i] += win[i + 2 * hop] * win[i + 2 * hop];
        steady[i] += win[i + hop] * win[i + hop];
        steady[i] += win[i] * win[i];
    }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) out[i] = ola[i + n_fft / 2] / (i < hop ? first[i] : steady[i % hop]);
}

void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    C::pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);

    // harmonic: sub-frame 2t uses envelope t, sub-frame 2t + 1 the mean of t and t + 1; spectra are linear in the cepstrum
    // so keep, per frame, the log magnitude and exp(i phase / 2), and build both sub-frames from frames t and t + 1
    const plan_t & hp = plan(harm_fft);
    struct half_t { float lr[257], hc[257], hs[257]; };
    half_t cur, nxt;
    int64_t have = -1;
    auto frame_spectrum = [&](int64_t t, half_t & h) {
        float cep[512] = {0}, li[257];
        const float * row = v + t * n_voc;
        for (int i = 0; i < n_cep_causal; i++) cep[i] = row[i] * ref::lifter(i, 0);
        for (int i = 0; i < n_cep_anti; i++) cep[harm_fft - 1 - i] = row[n_cep_causal + i] * ref::lifter(i, 1);
        C::rfft(hp, cep, h.lr, li);
        for (int k = 0; k <= hp.m; k++) { h.hc[k] = std::cos(0.5f * li[k]); h.hs[k] = std::sin(0.5f * li[k]); }
    };
    auto harm_response = [&](int64_t j, float * lr, float * pc, float * ps) {
        const int64_t s = std::min(j, harm_steps * n_frames - 1);
        const int64_t t = s / harm_steps, u = std::min(t + 1, n_frames - 1);
        if (have != t) {
            if (have == t - 1 && t - 1 >= 0) cur = nxt; else frame_spectrum(t, cur);
            frame_spectrum(u, nxt);
            have = t;
        }
        const half_t & b = s % harm_steps == 0 ? cur : nxt;
        for (int k = 0; k <= hp.m; k++) {
            lr[k] = 0.5f * (cur.lr[k] + b.lr[k]);
            pc[k] = cur.hc[k] * b.hc[k] - cur.hs[k] * b.hs[k];
            ps[k] = cur.hs[k] * b.hc[k] + cur.hc[k] * b.hs[k];
        }
    };
    auto harm_g = [&](int64_t j) { return harm_gain(v, std::min(j, harm_steps * n_frames - 1)); };

    const plan_t & np = plan(noise_fft);
    float nlr[129], npc[129], nps[129];
    int64_t nhave = -1;
    auto noise_response = [&](int64_t j, float * lr, float * pc, float * ps) {
        const int64_t t = std::min(j, noise_steps * n_frames - 1) / noise_steps;
        if (nhave != t) {
            float cep[256] = {0}, li[129];
            noise_envelope(v, t, cep);
            C::rfft(np, cep, nlr, li);
            for (int k = 0; k <= np.m; k++) { npc[k] = std::cos(li[k]); nps[k] = std::sin(li[k]); }
            nhave = t;
        }
        std::copy(nlr, nlr + np.m + 1, lr); std::copy(npc, npc + np.m + 1, pc); std::copy(nps, nps + np.m + 1, ps);
    };
    auto noise_g = [&](int64_t j) { return noise_gain(v, std::min(j, noise_steps * n_frames - 1)); };

    filter(pulses, hp, harm_hop, harm_response, harm_g, harm_out);
    filter(noise, np, noise_hop, noise_response, noise_g, noise_out);
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace D

// ------------------------------------------------------------------ T: C with the noise branch on a second thread
#include <thread>
namespace T {
void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    std::thread worker([&] {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto & x : noise) x = dist(rng);
        auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
        C::filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
                  [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    });
    C::pulse_train(f0, pulses);
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    C::filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    worker.join();
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace T

// ------------------------------------------------------------------ K: T + harmonic STFT frames split in chunks over threads
namespace K {
static int n_chunks = 2;
template <typename Key, typename Env, typename Gain>
static void filter_chunked(const std::vector<float> & src, int n_fft, int hop, Key key, Env env_cep, Gain gain, std::vector<float> & out, int chunks) {
    const B::plan_t & p = B::plan(n_fft);
    const int     n_bins = p.m + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;
    const int64_t n_stft = 1 + n_pad / hop;
    const int64_t n_full = n_pad + n_fft;
    const float * win = p.window.data();
    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);
    std::vector<std::vector<float>> olas(chunks);
    auto run = [&](int c) {
        const int64_t j0 = n_stft * c / chunks, j1 = n_stft * (c + 1) / chunks;
        std::vector<float> & ola = olas[c];
        ola.assign((size_t) ((j1 - j0) * hop + n_fft), 0.0f);
        float cbuf[512], fin[512], y[512], lr[257], li[257], pc[257], ps[257], sr[257], si[257];
        int64_t last_key = -1;
        for (int64_t j = j0; j < j1; j++) {
            const int64_t k_env = key(j);
            if (k_env != last_key) {
                std::fill(cbuf, cbuf + n_fft, 0.0f);
                env_cep(k_env, cbuf);
                C::rfft(p, cbuf, lr, li);
                for (int k = 0; k < n_bins; k++) { pc[k] = std::cos(li[k]); ps[k] = std::sin(li[k]); }
                last_key = k_env;
            }
            const float g = gain(j);
            const float * seg = x.data() + j * hop;
            for (int i = 0; i < n_fft; i++) fin[i] = seg[i] * win[i];
            C::rfft(p, fin, sr, si);
            for (int k = 0; k < n_bins; k++) {
                const float mag = std::exp(std::clamp(lr[k] + g, -14.0f, 6.0f));
                const float cc = mag * pc[k], d = mag * ps[k], a = sr[k], b = si[k];
                sr[k] = a * cc - b * d; si[k] = a * d + b * cc;
            }
            C::irfft(p, sr, si, y);
            float * o = ola.data() + (j - j0) * hop;
            for (int i = 0; i < n_fft; i++) o[i] += y[i] * win[i];
        }
    };
    std::vector<std::thread> ts;
    for (int c = 1; c < chunks; c++) ts.emplace_back(run, c);
    run(0);
    for (auto & t : ts) t.join();
    std::vector<float> ola((size_t) n_full, 0.0f);
    for (int c = 0; c < chunks; c++) {
        const int64_t j0 = n_stft * c / chunks;
        float * o = ola.data() + j0 * hop;
        for (size_t i = 0; i < olas[c].size(); i++) o[i] += olas[c][i];
    }
    float first[256], steady[256];
    for (int i = 0; i < hop; i++) {
        const float w0 = win[i] * win[i], w1 = win[i + hop] * win[i + hop], w2 = win[i + 2 * hop] * win[i + 2 * hop], w3 = win[i + 3 * hop] * win[i + 3 * hop];
        first[i] = (w2 + w1) + w0; steady[i] = ((w3 + w2) + w1) + w0;
    }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) out[i] = ola[i + n_fft / 2] / (i < hop ? first[i] : steady[i % hop]);
}
void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    std::thread worker([&] {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto & x : noise) x = dist(rng);
        auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
        C::filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
                  [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    });
    C::pulse_train(f0, pulses);
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    filter_chunked(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out, n_chunks);
    worker.join();
    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) audio[i] = harm_out[i] + noise_out[i];
}
}  // namespace K

namespace B {
void synth_profile(const std::vector<float> & voc, const std::vector<float> & f0, double * t) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::mt19937 rng(7);
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    auto t0 = clk::now();
    ref::pulse_train(f0, pulses);
    auto t1 = clk::now();
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);
    auto t2 = clk::now();
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    auto t3 = clk::now();
    filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
           [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    auto t4 = clk::now();
    t[0] += ms(t0, t1); t[1] += ms(t1, t2); t[2] += ms(t2, t3); t[3] += ms(t3, t4);
}
}
namespace C {
void synth_profile(const std::vector<float> & voc, const std::vector<float> & f0, double * t) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const int64_t n_frames = (int64_t) f0.size();
    const float * v = voc.data();
    std::mt19937 rng(7);
    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    auto t0 = clk::now();
    C::pulse_train(f0, pulses);
    auto t1 = clk::now();
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : noise) x = dist(rng);
    auto t2 = clk::now();
    auto hs = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto ns = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    C::filter(pulses, harm_fft, harm_hop, hs, [&](int64_t s, float * b) { harm_envelope(v, n_frames, s, b); }, [&](int64_t j) { return harm_gain(v, hs(j)); }, harm_out);
    auto t3 = clk::now();
    C::filter(noise, noise_fft, noise_hop, [&](int64_t j) { return ns(j) / noise_steps; }, [&](int64_t t, float * b) { noise_envelope(v, t, b); },
           [&](int64_t j) { return noise_gain(v, ns(j)); }, noise_out);
    auto t4 = clk::now();
    t[0] += ms(t0, t1); t[1] += ms(t1, t2); t[2] += ms(t2, t3); t[3] += ms(t3, t4);
}
}

static void C_profile(const std::vector<float> & v, const std::vector<float> & f, double * t) { C::synth_profile(v, f, t); }
// ------------------------------------------------------------------ driver
using synth_fn = void (*)(const std::vector<float> &, const std::vector<float> &, std::mt19937 &, std::vector<float> &);

static double snr_db(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0, e = 0;
    for (size_t i = 0; i < a.size(); i++) { s += (double) a[i] * a[i]; e += ((double) a[i] - b[i]) * ((double) a[i] - b[i]); }
    return e == 0 ? INFINITY : 10 * std::log10(s / e);
}

static double time_ms(synth_fn f, const std::vector<float> & voc, const std::vector<float> & f0, int reps) {
    std::vector<double> t;
    std::vector<float> out;
    for (int r = 0; r < reps; r++) {
        std::mt19937 rng(7);
        auto t0 = std::chrono::steady_clock::now();
        f(voc, f0, rng, out);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char ** argv) {
    if (getenv("THREADS")) {
        for (int a = 1; a < argc; a++) {
            FILE * fp = fopen(argv[a], "rb"); int64_t nf = 0; if (fread(&nf, 8, 1, fp) != 1) return 1;
            std::vector<float> f0(nf), voc(nf * n_voc);
            if (fread(f0.data(), 4, nf, fp) != (size_t) nf || fread(voc.data(), 4, voc.size(), fp) != voc.size()) return 1; fclose(fp);
            std::vector<float> c, t; { std::mt19937 rng(7); C::synth(voc, f0, rng, c); }
            const int reps = nf > 500 ? 15 : 41;
            const double tc = time_ms(C::synth, voc, f0, reps), tt = time_ms(T::synth, voc, f0, reps);
            printf("%s frames=%lld C=%.2f ms  T(2 thr)=%.2f (%+.0f%%)", strrchr(argv[a], '/') + 1, (long long) nf, tc, tt, 100 * (tt / tc - 1));
            for (int k : {2, 3}) {
                K::n_chunks = k;
                { std::mt19937 rng(7); K::synth(voc, f0, rng, t); }
                const double tk = time_ms(K::synth, voc, f0, reps);
                printf("  K(%d thr)=%.2f (%+.0f%%, SNR vs C %.0f dB)", k + 1, tk, 100 * (tk / tc - 1), snr_db(c, t));
            }
            printf("\n");
        }
        return 0;
    }
    if (getenv("PROFILE")) {
        for (int a = 1; a < argc; a++) {
            FILE * fp = fopen(argv[a], "rb"); int64_t nf = 0; if (fread(&nf, 8, 1, fp) != 1) return 1;
            std::vector<float> f0(nf), voc(nf * n_voc);
            if (fread(f0.data(), 4, nf, fp) != (size_t) nf || fread(voc.data(), 4, voc.size(), fp) != voc.size()) return 1; fclose(fp);
            double t[4] = {0, 0, 0, 0}; const int reps = 15;
            for (int r = 0; r < reps; r++) C_profile(voc, f0, t);
            const double tot = t[0] + t[1] + t[2] + t[3];
            printf("%s frames=%lld pulse=%.1f%% rng=%.1f%% harm=%.1f%% noise=%.1f%% total=%.2f ms\n", argv[a], (long long) nf, 100*t[0]/tot, 100*t[1]/tot, 100*t[2]/tot, 100*t[3]/tot, tot/reps);
        }
        return 0;
    }
    std::vector<std::pair<std::string, std::pair<std::vector<float>, std::vector<float>>>> cases;
    for (int a = 1; a < argc; a++) {
        FILE * fp = fopen(argv[a], "rb");
        int64_t nf = 0;
        if (!fp || fread(&nf, sizeof(nf), 1, fp) != 1) return 1;
        std::vector<float> f0(nf), voc(nf * n_voc);
        if (fread(f0.data(), 4, nf, fp) != (size_t) nf || fread(voc.data(), 4, voc.size(), fp) != voc.size()) return 1;
        fclose(fp);
        cases.push_back({argv[a], {voc, f0}});
    }
    // edge cases: 1 and 2 frames cut from the first utterance
    for (int nf : {1, 2, 3}) {
        auto & [voc, f0] = cases[0].second;
        cases.push_back({"cut" + std::to_string(nf), {std::vector<float>(voc.begin(), voc.begin() + nf * n_voc), std::vector<float>(f0.begin(), f0.begin() + nf)}});
    }
    printf("%-10s %6s | %8s %8s %8s %8s | %7s %7s %7s | %6s %6s %6s\n", "case", "frames", "ref_ms", "A_ms", "B_ms", "C_ms", "SNR_A", "SNR_B", "SNR_C", "A", "B", "C");
    printf("(D printed on a second line per case)\n");
    for (auto & [name, data] : cases) {
        auto & [voc, f0] = data;
        std::vector<float> r, a, b, c, pulses;
        { std::mt19937 rng(7); ref::synth(voc, f0, rng, r); }
        { std::mt19937 rng(7); A::synth(voc, f0, rng, a); }
        { std::mt19937 rng(7); B::synth(voc, f0, rng, b); }
        { std::mt19937 rng(7); C::synth(voc, f0, rng, c); }
        std::vector<float> d; { std::mt19937 rng(7); D::synth(voc, f0, rng, d); }
        const int reps = f0.size() > 500 ? 9 : 31;
        std::vector<double> tp;
        for (int i = 0; i < reps; i++) { auto t0 = std::chrono::steady_clock::now(); ref::pulse_train(f0, pulses); tp.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
        std::sort(tp.begin(), tp.end());
        const double tr = time_ms(ref::synth, voc, f0, reps), ta = time_ms(A::synth, voc, f0, reps), tb = time_ms(B::synth, voc, f0, reps), tc = time_ms(C::synth, voc, f0, reps);
        std::string base = name.substr(name.find_last_of('/') + 1);
        printf("%-10s %6zu | %8.3f %8.3f %8.3f %8.3f | %7.1f %7.1f %7.1f | %5.0f%% %5.0f%% %5.0f%%\n", base.c_str(), f0.size(), tr, ta, tb, tc,
               snr_db(r, a), snr_db(r, b), snr_db(r, c), 100 * (1 - ta / tr), 100 * (1 - tb / tr), 100 * (1 - tc / tr));
        const double td = time_ms(D::synth, voc, f0, reps);
        std::vector<float> tt; { std::mt19937 rng(7); T::synth(voc, f0, rng, tt); }
        const double tth = time_ms(T::synth, voc, f0, reps);
        printf("%-10s %6s   T_ms %8.3f   SNR_T vs C %s   (T vs C %+.0f%%)\n", "", "", tth, snr_db(c, tt) == INFINITY ? "bit-identical" : "DIFF", 100 * (tth / tc - 1));
        printf("%-10s %6s   D_ms %8.3f   SNR_D %7.1f   D %5.0f%%   (D vs C %+.0f%%)\n", "", "", td, snr_db(r, d), 100 * (1 - td / tr), 100 * (td / tc - 1));
    }
}
