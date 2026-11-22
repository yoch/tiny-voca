#include "models.h"
#include "../mtmd-audio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>

// same-length conv with 3 taps; x: [T, IC], w: [3, IC, OC], returns [T, OC]
ggml_tensor * mtmd_graph_tiny_voca::conv3(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
    ggml_tensor * col = ggml_im2col(ctx0, w, x, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx0,
            ggml_reshape_2d(ctx0, col, col->ne[0], col->ne[1] * col->ne[2]),
            ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]));
    return ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
}

ggml_cgraph * mtmd_graph_tiny_voca::build() {
    const auto & tv = model.tiny_voca;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_rows);
    ggml_set_name(inp, "inp_feats");
    ggml_set_input(inp);

    ggml_tensor * cur;
    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_IN) {
        // duration predictor: (conv, gelu, layer norm) x 2, then a linear layer
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        cur = conv3(cur, tv.dur_conv1_w, tv.dur_conv1_b);
        cur = ggml_gelu_erf(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
        cur = build_norm(cur, tv.dur_norm1_w, tv.dur_norm1_b, NORM_TYPE_NORMAL, eps, -1);
        cur = conv3(ggml_cont(ctx0, ggml_transpose(ctx0, cur)), tv.dur_conv2_w, tv.dur_conv2_b);
        cur = ggml_gelu_erf(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
        cur = build_norm(cur, tv.dur_norm2_w, tv.dur_norm2_b, NORM_TYPE_NORMAL, eps, -1);
        cur = ggml_add(ctx0, build_mm(tv.dur_out_w, cur), tv.dur_out_b);
    } else {
        GGML_ASSERT(gen_process == MTMD_SUBCTX_GEN_PROCESS_LATENT_OUT);
        ggml_tensor * h = ggml_add(ctx0, build_mm(tv.harm_w,  inp), tv.harm_b);
        ggml_tensor * n = ggml_add(ctx0, build_mm(tv.noise_w, inp), tv.noise_b);
        cur = ggml_concat(ctx0, h, n, 0);
    }
    ggml_set_name(cur, "out_feats");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

//
// host: harmonic plus noise synthesis, see SourceFilter in the reference (model.py)
//

// band-limited pulse train with unit power, F0 given per frame and interpolated in the log domain
static void tiny_voca_pulse_train(const std::vector<float> & f0, std::vector<float> & out) {
    using g = mtmd_graph_tiny_voca;
    const int64_t n_frames = (int64_t) f0.size();
    std::vector<double> knots(n_frames + 1);
    for (int64_t i = 0; i < n_frames; i++) {
        knots[i] = std::log(std::max((double) f0[i], 1.0));
    }
    knots[n_frames] = knots[n_frames - 1];

    out.resize((size_t) (n_frames * g::hop));
    double acc = 0.0; // phase in cycles, kept in [0, 1)
    for (int64_t i = 0; i < n_frames; i++) {
        double freq = std::exp(knots[i]);
        const double ratio = std::exp((knots[i + 1] - knots[i]) / g::hop);
        for (int m = 0; m < g::hop; m++, freq *= ratio) {
            acc += freq / g::sample_rate;
            acc -= std::floor(acc);
            const double theta = 2.0 * M_PI * acc;
            const double harmonics = std::max((double) g::harmonic_limit * g::sample_rate / freq, 1.0);
            const double whole = std::floor(harmonics);
            // cos((whole + 1) theta) from the angles below, so only two sin/cos pairs per sample
            const double h = theta / 2.0;
            const double a = (whole + 0.5) * theta;
            const double sh = std::sin(h), ch = std::cos(h), sa = std::sin(a), ca = std::cos(a);
            const double core = std::fabs(2.0 * sh) < 1e-7 ? whole : sa / (2.0 * sh) - 0.5;
            const double wave = core + (harmonics - whole) * (ca * ch - sa * sh);
            out[i * g::hop + m] = (float) (wave * std::sqrt(2.0 / harmonics));
        }
    }
}

// real FFT of n samples as a complex FFT of n / 2 points, for n = 256 or 512
struct tiny_voca_rfft {
    int n, m;
    std::vector<int>   rev;
    std::vector<float> tw_c, tw_s, rot_c, rot_s, win;

    explicit tiny_voca_rfft(int n) : n(n), m(n / 2), rev(m), tw_c(m), tw_s(m), rot_c(m + 1), rot_s(m + 1), win(n) {
        for (int k = 0; k < m; k++) {
            int r = 0;
            for (int b = m, i = k; b > 1; b >>= 1, i >>= 1) {
                r = 2 * r + (i & 1);
            }
            rev[k] = r;
        }
        for (int half = 1; half < m; half *= 2) {
            for (int j = 0; j < half; j++) {
                tw_c[half - 1 + j] = std::cos(-M_PI * j / half);
                tw_s[half - 1 + j] = std::sin(-M_PI * j / half);
            }
        }
        for (int k = 0; k <= m; k++) {
            rot_c[k] = std::cos(-2.0 * M_PI * k / n);
            rot_s[k] = std::sin(-2.0 * M_PI * k / n);
        }
        for (int i = 0; i < n; i++) {
            win[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / n));
        }
    }

    void butterflies(float * r, float * im) const {
        // stages of size 2 and 4 together, their twiddles are 1 and -i
        for (int a = 0; a < m; a += 4) {
            const float r0 = r[a] + r[a + 1],     i0 = im[a] + im[a + 1],     r1 = r[a] - r[a + 1],     i1 = im[a] - im[a + 1];
            const float r2 = r[a + 2] + r[a + 3], i2 = im[a + 2] + im[a + 3], r3 = r[a + 2] - r[a + 3], i3 = im[a + 2] - im[a + 3];
            r[a]     = r0 + r2; im[a]     = i0 + i2;
            r[a + 2] = r0 - r2; im[a + 2] = i0 - i2;
            r[a + 1] = r1 + i3; im[a + 1] = i1 - r3;
            r[a + 3] = r1 - i3; im[a + 3] = i1 + r3;
        }
        for (int half = 4; half < m; half *= 2) {
            const float * c = tw_c.data() + half - 1;
            const float * s = tw_s.data() + half - 1;
            for (int block = 0; block < m; block += 2 * half) {
                for (int j = 0; j < half; j++) {
                    const int   a  = block + j, b = a + half;
                    const float tr = r[b] * c[j] - im[b] * s[j];
                    const float ti = r[b] * s[j] + im[b] * c[j];
                    r[b] = r[a] - tr; im[b] = im[a] - ti;
                    r[a] += tr;       im[a] += ti;
                }
            }
        }
    }

    // x: n samples -> m + 1 bins
    void forward(const float * x, float * out_r, float * out_i) const {
        float r[256], im[256];
        for (int j = 0; j < m; j++) {
            r[j]  = x[2 * rev[j]];
            im[j] = x[2 * rev[j] + 1];
        }
        butterflies(r, im);
        out_r[0] = r[0] + im[0];
        out_r[m] = r[0] - im[0];
        out_i[0] = out_i[m] = 0.0f;
        for (int k = 1; k < m; k++) {
            const float dr = r[k] - r[m - k], di = im[k] + im[m - k];
            out_r[k] = 0.5f * (r[k] + r[m - k] + di * rot_c[k] + dr * rot_s[k]);
            out_i[k] = 0.5f * (im[k] - im[m - k] - dr * rot_c[k] + di * rot_s[k]);
        }
    }

    // m + 1 bins -> n samples, scaled by 1 / n; the imaginary parts of DC and Nyquist are ignored
    void inverse(const float * x_r, const float * x_i, float * y) const {
        float r[256], im[256];
        for (int j = 0; j < m; j++) {
            const int   k  = rev[j], o = m - k;
            const float dr = x_r[k] - x_r[o], di = x_i[k] + x_i[o];
            r[j]  =  0.5f * (x_r[k] + x_r[o] - di * rot_c[k] + dr * rot_s[k]);
            im[j] = -0.5f * (x_i[k] - x_i[o] + dr * rot_c[k] + di * rot_s[k]);
        }
        butterflies(r, im);
        for (int j = 0; j < m; j++) {
            y[2 * j]     =  r[j] / m;
            y[2 * j + 1] = -im[j] / m;
        }
    }
};

// time-varying filter: STFT of the source (center, zero padding), times exp(rfft(cepstrum) + gain) of each frame, ISTFT
// env(e, buf) writes the n_fft cepstrum of envelope e without the gain; key(j) and gain(j) give them for STFT frame j
// the source gets n_fft / 2 trailing zeros first
static void tiny_voca_filter(const std::vector<float> & src, const tiny_voca_rfft & p, int hop,
        const std::function<int64_t(int64_t)> & key, const std::function<void(int64_t, float *)> & env_cep,
        const std::function<float(int64_t)> & gain, std::vector<float> & out) {
    const int     n_fft  = p.n;
    const int     n_bins = p.m + 1;
    const int64_t n_src  = (int64_t) src.size();
    const int64_t n_pad  = n_src + n_fft / 2;      // length given to the STFT
    const int64_t n_stft = 1 + n_pad / hop;        // frames, center = true
    const int64_t n_full = n_pad + n_fft;          // with the center padding
    const float * win = p.win.data();

    std::vector<float> x((size_t) n_full, 0.0f);
    std::copy(src.begin(), src.end(), x.begin() + n_fft / 2);

    std::vector<float> ola((size_t) n_full, 0.0f), env((size_t) n_full, 0.0f);
    float cbuf[512], fin[512], y[512], lr[257], li[257], pc[257], ps[257], sr[257], si[257];
    int64_t last = -1;
    for (int64_t j = 0; j < n_stft; j++) {
        // response of this frame; the envelope is the same for several frames, only the gain changes
        const int64_t e = key(j);
        if (e != last) {
            std::fill(cbuf, cbuf + n_fft, 0.0f);
            env_cep(e, cbuf);
            p.forward(cbuf, lr, li);
            for (int k = 0; k < n_bins; k++) {
                pc[k] = std::cos(li[k]);
                ps[k] = std::sin(li[k]);
            }
            last = e;
        }
        const float g = gain(j);

        // filtered spectrum of the source frame
        const float * seg = x.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            fin[i] = seg[i] * win[i];
        }
        p.forward(fin, sr, si);
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(lr[k] + g, -14.0f, 6.0f));
            const float a = sr[k], b = si[k], c = mag * pc[k], d = mag * ps[k];
            sr[k] = a * c - b * d;
            si[k] = a * d + b * c;
        }
        p.inverse(sr, si, y);

        float * o = ola.data() + j * hop;
        float * ev = env.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            o[i] += y[i] * win[i];
            ev[i] += win[i] * win[i];
        }
    }

    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) {
        const int64_t k = i + n_fft / 2; // drop the center padding
        out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f;
    }
}

void mtmd_graph_tiny_voca::synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio) {
    const int64_t n_frames = (int64_t) f0.size();
    GGML_ASSERT(n_frames > 0 && voc.size() == (size_t) (n_frames * n_voc));
    static_assert(harm_fft == 512 && noise_fft == 256, "tiny_voca_rfft buffers hold n_fft <= 512");
    static const tiny_voca_rfft harm_plan(harm_fft), noise_plan(noise_fft);

    // liftering: coefficient i scaled by 1 / (1 + i / 4), the gain (coefficient 0) by 2
    auto lifter = [](int i, int first) { return i == 0 && first == 0 ? 2.0f : 1.0f / (1.0f + (float) (i + first) / 4.0f); };

    // harmonic branch: envelope interpolated between frames, one gain per sub-frame
    const int split = n_cep_causal + n_cep_anti;
    auto harm_sub = [&](int64_t j) { return std::min(j, harm_steps * n_frames - 1); };
    auto harm_env = [&](int64_t s, float * buf) {
        const float * here = voc.data() + (s / harm_steps) * n_voc;
        const float * next = voc.data() + std::min((s + 1) / harm_steps, n_frames - 1) * n_voc;
        for (int i = 0; i < n_cep_causal; i++) {
            buf[i] = 0.5f * (here[i] + next[i]) * lifter(i, 0);
        }
        for (int i = 0; i < n_cep_anti; i++) {
            buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * lifter(i, 1);
        }
    };
    auto harm_gain = [&](int64_t j) {
        const int64_t s = harm_sub(j);
        return voc[(s / harm_steps) * n_voc + split + s % harm_steps];
    };
    // noise branch: one envelope per frame, one gain per sub-frame
    auto noise_sub = [&](int64_t j) { return std::min(j, noise_steps * n_frames - 1); };
    auto noise_env = [&](int64_t t, float * buf) {
        const float * row = voc.data() + t * n_voc + split + harm_steps;
        for (int i = 0; i < n_noise_cep; i++) {
            buf[i] = row[i] * lifter(i, 0);
        }
    };
    auto noise_gain = [&](int64_t j) {
        const int64_t s = noise_sub(j);
        return voc[(s / noise_steps) * n_voc + split + harm_steps + n_noise_cep + s % noise_steps];
    };

    std::vector<float> pulses, noise((size_t) (n_frames * hop)), harm_out, noise_out;
    tiny_voca_pulse_train(f0, pulses);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : noise) {
        v = dist(rng);
    }
    tiny_voca_filter(pulses, harm_plan,  harm_hop,  harm_sub, harm_env, harm_gain, harm_out);
    tiny_voca_filter(noise,  noise_plan, noise_hop, [&](int64_t j) { return noise_sub(j) / noise_steps; }, noise_env, noise_gain, noise_out);

    audio.resize(pulses.size());
    for (size_t i = 0; i < audio.size(); i++) {
        audio[i] = harm_out[i] + noise_out[i];
    }
}

//
// host: unknown-word model of the phonemizer, see g2p.py of dizione
//

static const char * const tiny_voca_g2p_letters = "abcdefghijklmnopqrstuvwxyz'";
static const char * const tiny_voca_g2p_symbols[mtmd_graph_tiny_voca::n_g2p_symbol] = {
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
};

// one direction of a GRU layer over x [n, n_in] -> y [n, n_hidden], written at column offset of rows of width stride
// the products are summed in double and rounded once, as close as can be to the reference (numpy, float32): two
// symbols are sometimes almost as likely, and the order of a float32 sum would decide between them
static void tiny_voca_gru(const mtmd_tiny_voca::g2p_gru & g, const std::vector<float> & x, int64_t n, int64_t n_in, int64_t n_hidden,
        bool reverse, std::vector<float> & y, int64_t offset, int64_t stride) {
    const int64_t h3 = 3 * n_hidden;
    std::vector<float> inp((size_t) (n * h3));
    for (int64_t t = 0; t < n; t++) {
        for (int64_t r = 0; r < h3; r++) {
            const float * w = g.w_ih.data() + r * n_in;
            const float * xt = x.data() + t * n_in;
            double acc = 0.0;
            for (int64_t c = 0; c < n_in; c++) {
                acc += (double) w[c] * xt[c];
            }
            inp[t * h3 + r] = (float) acc + g.b_ih[r];
        }
    }
    std::vector<float> state((size_t) n_hidden, 0.0f), rec((size_t) h3);
    for (int64_t s = 0; s < n; s++) {
        const int64_t t = reverse ? n - 1 - s : s;
        for (int64_t r = 0; r < h3; r++) {
            const float * w = g.w_hh.data() + r * n_hidden;
            double acc = 0.0;
            for (int64_t c = 0; c < n_hidden; c++) {
                acc += (double) w[c] * state[c];
            }
            rec[r] = (float) acc + g.b_hh[r];
        }
        const float * in = inp.data() + t * h3;
        for (int64_t k = 0; k < n_hidden; k++) {
            const float reset  = 1.0f / (1.0f + std::exp(-(in[k] + rec[k])));
            const float update = 1.0f / (1.0f + std::exp(-(in[n_hidden + k] + rec[n_hidden + k])));
            const float cand   = std::tanh(in[2 * n_hidden + k] + reset * rec[2 * n_hidden + k]);
            state[k] = (1.0f - update) * cand + update * state[k];
        }
        std::copy(state.begin(), state.end(), y.begin() + t * stride + offset);
    }
}

std::string mtmd_graph_tiny_voca::g2p(const mtmd_tiny_voca & tv, const std::string & word) {
    const int64_t w = tv.g2p_n_width;
    const int64_t h = tv.g2p_n_hidden;
    if (w == 0 || tv.g2p_layers.empty()) {
        return "";
    }
    std::vector<int> ids;
    for (char c : word) {
        const char * p = strchr(tiny_voca_g2p_letters, std::tolower((unsigned char) c));
        if (c != '\0' && p) {
            ids.push_back((int) (p - tiny_voca_g2p_letters) + 1);
        }
    }
    if (ids.empty()) {
        return "";
    }

    // each letter takes n_g2p_slot positions: its embedding plus the embedding of the slot
    const int64_t n = (int64_t) ids.size() * n_g2p_slot;
    std::vector<float> x((size_t) (n * w));
    for (size_t l = 0; l < ids.size(); l++) {
        for (int s = 0; s < n_g2p_slot; s++) {
            float * row = x.data() + ((int64_t) l * n_g2p_slot + s) * w;
            for (int64_t c = 0; c < w; c++) {
                row[c] = tv.g2p_embd[ids[l] * w + c] + tv.g2p_slot[s * w + c];
            }
        }
    }
    int64_t n_in = w;
    for (size_t il = 0; il + 1 < tv.g2p_layers.size(); il += 2) {
        std::vector<float> y((size_t) (n * 2 * h));
        tiny_voca_gru(tv.g2p_layers[il],     x, n, n_in, h, false, y, 0, 2 * h);
        tiny_voca_gru(tv.g2p_layers[il + 1], x, n, n_in, h, true,  y, h, 2 * h);
        x    = std::move(y);
        n_in = 2 * h;
    }

    // CTC, read greedily: a label repeated or the blank (0) is no new symbol
    std::string out;
    int prev = 0;
    for (int64_t t = 0; t < n; t++) {
        int best = 0;
        float best_v = -INFINITY;
        for (int k = 0; k <= n_g2p_symbol; k++) {
            const float * wr = tv.g2p_out_w.data() + k * n_in;
            double sum = 0.0;
            for (int64_t c = 0; c < n_in; c++) {
                sum += (double) wr[c] * x[t * n_in + c];
            }
            const float acc = (float) sum + tv.g2p_out_b[k];
            if (acc > best_v) {
                best_v = acc;
                best   = k;
            }
        }
        if (best != prev && best != 0) {
            out += (out.empty() ? "" : " ") + std::string(tiny_voca_g2p_symbols[best - 1]);
        }
        prev = best;
    }
    return out;
}
