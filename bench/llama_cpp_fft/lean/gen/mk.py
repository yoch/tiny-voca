# composes candidate tiny-voca.cpp files from the original by exact text replacements, one function per idea
import sys
ORIG = open(sys.argv[1]).read()

def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)

# ---------------------------------------------------------------- pulse train
PULSE_OLD = """    double acc = 0.0;
    for (int64_t n = 0; n < n_frames * g::hop; n++) {
        const int64_t low = n / g::hop;
        const double  w   = (double) (n - low * g::hop) / g::hop;
        const double  freq = std::exp(knots[low] * (1.0 - w) + knots[low + 1] * w);
        acc += freq / g::sample_rate;
        const double theta = std::fmod(2.0 * M_PI * acc, 2.0 * M_PI);
        const double harmonics = std::max((double) g::harmonic_limit * g::sample_rate / freq, 1.0);
        const double whole = std::floor(harmonics);
        const double denom = 2.0 * std::sin(theta / 2.0);
        const double core  = std::fabs(denom) < 1e-7 ? whole : std::sin((whole + 0.5) * theta) / denom - 0.5;
        const double wave  = core + (harmonics - whole) * std::cos((whole + 1.0) * theta);
        out[n] = (float) (wave * std::sqrt(2.0 / harmonics));
    }
"""
def pulse_C(s):  # variant C pulse train: one exp per frame, phase in cycles, two sin/cos pairs
    c = open(sys.argv[2]).read()
    a = c.index("    double acc = 0.0; // phase in cycles")
    b = c.index("\n}\n", a)
    return rep(s, PULSE_OLD, c[a:b] + "\n")

def pulse_wrap(s):  # phase in cycles wrapped with floor instead of fmod of an ever-growing phase
    s = rep(s, "    double acc = 0.0;\n    for (int64_t n", "    double acc = 0.0; // phase in cycles, kept in [0, 1)\n    for (int64_t n")
    return rep(s, """        acc += freq / g::sample_rate;
        const double theta = std::fmod(2.0 * M_PI * acc, 2.0 * M_PI);
""", """        acc += freq / g::sample_rate;
        acc -= std::floor(acc);
        const double theta = 2.0 * M_PI * acc;
""")

def pulse_exp(s):  # one exp per frame: exp of the linear log-F0 ramp is a geometric sequence
    return rep(s, """    for (int64_t n = 0; n < n_frames * g::hop; n++) {
        const int64_t low = n / g::hop;
        const double  w   = (double) (n - low * g::hop) / g::hop;
        const double  freq = std::exp(knots[low] * (1.0 - w) + knots[low + 1] * w);
""", """    for (int64_t n = 0; n < n_frames * g::hop; n++) {
        const int64_t low = n / g::hop;
        const double  w   = (double) (n - low * g::hop) / g::hop;
        const double  freq = std::exp(knots[low] * (1.0 - w) + knots[low + 1] * w);
""".replace("XX", ""))  # placeholder, replaced below

def pulse_trig(s):  # cos((whole + 1) theta) from the two angles already needed: two sin/cos pairs per sample
    return rep(s, """        const double denom = 2.0 * std::sin(theta / 2.0);
        const double core  = std::fabs(denom) < 1e-7 ? whole : std::sin((whole + 0.5) * theta) / denom - 0.5;
        const double wave  = core + (harmonics - whole) * std::cos((whole + 1.0) * theta);
""", """        // cos((whole + 1) theta) = cos(a + h) with h = theta / 2, a = (whole + 0.5) theta: two sin/cos pairs per sample
        const double sh = std::sin(theta / 2.0), ch = std::cos(theta / 2.0);
        const double sa = std::sin((whole + 0.5) * theta), ca = std::cos((whole + 0.5) * theta);
        const double core  = std::fabs(2.0 * sh) < 1e-7 ? whole : sa / (2.0 * sh) - 0.5;
        const double wave  = core + (harmonics - whole) * (ca * ch - sa * sh);
""")

# ---------------------------------------------------------------- filter: tables once
def tables_once(s):  # sin/cos table and Hann window built once per FFT size
    return rep(s, """    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    const float * win = cache.hann_window.data();
""", """    // sin/cos table and Hann window built once per FFT size
    static const auto make_cache = [](int n) {
        mtmd_audio_cache c;
        c.fill_sin_cos_table(n);
        c.fill_hann_window(n, true);
        return c;
    };
    static const mtmd_audio_cache cache_harm = make_cache(mtmd_graph_tiny_voca::harm_fft), cache_noise = make_cache(mtmd_graph_tiny_voca::noise_fft);
    const mtmd_audio_cache & cache = n_fft == mtmd_graph_tiny_voca::harm_fft ? cache_harm : cache_noise;
    GGML_ASSERT((int) cache.sin_vals.size() == n_fft);
    const float * win = cache.hann_window.data();
""")

# ---------------------------------------------------------------- filter: normalization precomputed
def norm(s):  # the sum of squared windows is periodic in hop: no env buffer
    s = rep(s, "    std::vector<float> ola((size_t) n_full, 0.0f), env((size_t) n_full, 0.0f);\n", "    std::vector<float> ola((size_t) n_full, 0.0f);\n")
    s = rep(s, """        float * o = ola.data() + j * hop;
        float * e = env.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            o[i] += rout[2 * i] * win[i];
            e[i] += win[i] * win[i];
        }
""", """        float * o = ola.data() + j * hop;
        for (int i = 0; i < n_fft; i++) {
            o[i] += rout[2 * i] * win[i];
        }
""")
    return rep(s, """    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) {
        const int64_t k = i + n_fft / 2; // drop the center padding
        out[i] = env[k] > 1e-11f ? ola[k] / env[k] : 0.0f;
    }
""", """    // sum of the squared windows, in frame order: three windows over the first kept hop, four after
    GGML_ASSERT(n_fft == 4 * hop);
    std::vector<float> first(hop), steady(hop);
    for (int i = 0; i < hop; i++) {
        const float w0 = win[i] * win[i], w1 = win[i + hop] * win[i + hop], w2 = win[i + 2 * hop] * win[i + 2 * hop], w3 = win[i + 3 * hop] * win[i + 3 * hop];
        first[i]  = (w2 + w1) + w0;
        steady[i] = ((w3 + w2) + w1) + w0;
    }
    out.resize((size_t) n_src);
    for (int64_t i = 0; i < n_src; i++) {
        out[i] = ola[i + n_fft / 2] / (i < hop ? first[i] : steady[i % hop]); // drop the center padding
    }
""")

# ---------------------------------------------------------------- filter: real FFT as a half-size complex FFT (existing mtmd_audio_ifft)
HALF_HELPERS = """// real FFT of n samples as a complex FFT of m = n / 2 points with mtmd_audio_ifft, the cache holds the sin/cos table of n
// split step, k = 0 .. m: z[k] = (x[k] + conj(x[m - k])) / 2 + i e^(2 pi i k / n) (x[k] - conj(x[m - k])) / 2
static void tiny_voca_split(const mtmd_audio_cache & cache, const float * x, int m, float * z) {
    for (int k = 0; k <= m; k++) {
        const float c  = cache.cos_vals[k], s = cache.sin_vals[k];
        const float er = 0.5f * (x[2 * k] + x[2 * (m - k)]),     ei = 0.5f * (x[2 * k + 1] - x[2 * (m - k) + 1]);
        const float dr = 0.5f * (x[2 * k] - x[2 * (m - k)]),     di = 0.5f * (x[2 * k + 1] + x[2 * (m - k) + 1]);
        z[2 * k]     = er - c * di - s * dr;
        z[2 * k + 1] = ei + c * dr - s * di;
    }
}

// x: n samples -> X: m + 1 bins; FFT(z) = m conj(IFFT(conj(z))) for z[j] = x[2 j] + i x[2 j + 1]; in and out: 4 n floats
static void tiny_voca_rfft(const mtmd_audio_cache & cache, const float * x, int n, float * in, float * out, float * X) {
    const int m = n / 2;
    for (int i = 0; i < n; i += 2) {
        in[i]     =  x[i];
        in[i + 1] = -x[i + 1];
    }
    mtmd_audio_ifft(cache, in, m, out);
    out[2 * m] = out[0];
    out[2 * m + 1] = out[1];
    tiny_voca_split(cache, out, m, X);
    for (int k = 0; k <= m; k++) {
        X[2 * k]     *=  m;
        X[2 * k + 1] *= -m;
    }
}

// X: m + 1 bins -> y: n samples, scaled by 1 / n; in and y: 4 n floats
static void tiny_voca_irfft(const mtmd_audio_cache & cache, const float * X, int n, float * in, float * y) {
    tiny_voca_split(cache, X, n / 2, in);
    mtmd_audio_ifft(cache, in, n / 2, y);
}

"""
def half(s):
    s = rep(s, "// time-varying filter: STFT", HALF_HELPERS + "// time-varying filter: STFT")
    s = rep(s, """    std::vector<float> fin(8 * n_fft), fout(8 * n_fft), rin(8 * n_fft), rout(8 * n_fft), cbuf(n_fft);
    std::vector<float> resp(2 * n_bins);
""", """    std::vector<float> fin(n_fft), fout(2 * n_bins), rin(4 * n_fft), rout(4 * n_fft), cbuf(n_fft);
    std::vector<float> resp(2 * n_bins);
""")
    s = rep(s, """        std::copy(cbuf.begin(), cbuf.end(), rin.begin());
        mtmd_audio_fft(cache, rin.data(), n_fft, rout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(rout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(rout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(rout[2 * k + 1]);
        }
""", """        tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), rout.data(), fout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(fout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(fout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(fout[2 * k + 1]);
        }
""")
    s = rep(s, """        mtmd_audio_fft(cache, fin.data(), n_fft, fout.data());

        // filtered spectrum, mirrored for the complex inverse FFT; irfft ignores the imaginary parts of DC and Nyquist
        for (int k = 0; k < n_bins; k++) {
            const float a = fout[2 * k], b = fout[2 * k + 1];
            const float c = resp[2 * k], d = resp[2 * k + 1];
            rin[2 * k + 0] = a * c - b * d;
            rin[2 * k + 1] = (k == 0 || k == n_fft / 2) ? 0.0f : a * d + b * c;
        }
        for (int k = 1; k < n_fft - n_bins + 1; k++) {
            rin[2 * (n_fft - k) + 0] =  rin[2 * k + 0];
            rin[2 * (n_fft - k) + 1] = -rin[2 * k + 1];
        }
        mtmd_audio_ifft(cache, rin.data(), n_fft, rout.data());
""", """        tiny_voca_rfft(cache, fin.data(), n_fft, rin.data(), rout.data(), fout.data());

        // filtered spectrum; irfft ignores the imaginary parts of DC and Nyquist
        for (int k = 0; k < n_bins; k++) {
            const float a = fout[2 * k], b = fout[2 * k + 1];
            const float c = resp[2 * k], d = resp[2 * k + 1];
            fout[2 * k + 0] = a * c - b * d;
            fout[2 * k + 1] = (k == 0 || k == n_fft / 2) ? 0.0f : a * d + b * c;
        }
        tiny_voca_irfft(cache, fout.data(), n_fft, rin.data(), rout.data());
""")
    # output is now n real samples, not interleaved complex
    s = rep(s, "            o[i] += rout[2 * i] * win[i];\n", "            o[i] += rout[i] * win[i];\n")
    return s


# ---------------------------------------------------------------- filter: response once per envelope, the gain added to the log magnitude
def resp_once(s):
    s = rep(s, """// cep(j, buf) writes the n_fft cepstrum of STFT frame j, the source gets n_fft / 2 trailing zeros first
static void tiny_voca_filter(const std::vector<float> & src, int n_fft, int hop,
        const std::function<void(int64_t, float *)> & cep, std::vector<float> & out) {""",
"""// cep(j, buf) writes the n_fft cepstrum of STFT frame j without the gain (coefficient 0) and returns the gain,
// the source gets n_fft / 2 trailing zeros first
static void tiny_voca_filter(const std::vector<float> & src, int n_fft, int hop,
        const std::function<float(int64_t, float *)> & cep, std::vector<float> & out) {""")
    s = rep(s, "    std::vector<float> resp(2 * n_bins);\n", "    std::vector<float> resp(2 * n_bins), prev, logmag(n_bins), pc(n_bins), ps(n_bins);\n")
    # the response block, with the original FFT or with the half-size one
    old_fft = """        std::fill(cbuf.begin(), cbuf.end(), 0.0f);
        cep(j, cbuf.data());
        std::copy(cbuf.begin(), cbuf.end(), rin.begin());
        mtmd_audio_fft(cache, rin.data(), n_fft, rout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(rout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(rout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(rout[2 * k + 1]);
        }
"""
    old_half = """        std::fill(cbuf.begin(), cbuf.end(), 0.0f);
        cep(j, cbuf.data());
        tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), rout.data(), fout.data());
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(fout[2 * k], -14.0f, 6.0f));
            resp[2 * k + 0] = mag * std::cos(fout[2 * k + 1]);
            resp[2 * k + 1] = mag * std::sin(fout[2 * k + 1]);
        }
"""
    new = """        std::fill(cbuf.begin(), cbuf.end(), 0.0f);
        const float gain = cep(j, cbuf.data());
        if (cbuf != prev) { // the envelope changes less often than the gain
            FFT_CALL
            for (int k = 0; k < n_bins; k++) {
                logmag[k] = SPEC[2 * k];
                pc[k] = std::cos(SPEC[2 * k + 1]);
                ps[k] = std::sin(SPEC[2 * k + 1]);
            }
            prev = cbuf;
        }
        for (int k = 0; k < n_bins; k++) {
            const float mag = std::exp(std::clamp(logmag[k] + gain, -14.0f, 6.0f));
            resp[2 * k + 0] = mag * pc[k];
            resp[2 * k + 1] = mag * ps[k];
        }
"""
    old_iter = old_half.replace("rin.data(), rout.data(), fout.data()", "rin.data(), fout.data()")
    if old_iter in s:  # I or J
        s = rep(s, old_iter, new.replace("FFT_CALL", "tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), fout.data());").replace("SPEC", "fout"))
    elif old_fft in s:
        s = rep(s, old_fft, new.replace("FFT_CALL", """std::copy(cbuf.begin(), cbuf.end(), rin.begin());
            mtmd_audio_fft(cache, rin.data(), n_fft, rout.data());""").replace("SPEC", "rout"))
    else:
        s = rep(s, old_half, new.replace("FFT_CALL", "tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), rout.data(), fout.data());").replace("SPEC", "fout"))
    return resp_tail(s)

def resp_tail(s):
    s = rep(s, "    auto harm_cep = [&](int64_t j, float * buf) {", "    auto harm_cep = [&](int64_t j, float * buf) -> float {")
    s = rep(s, """        buf[0] += here[split + s % harm_steps];
        for (int i = 0; i < n_cep_anti; i++) {
            buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * lifter(i, 1);
        }
""", """        for (int i = 0; i < n_cep_anti; i++) {
            buf[harm_fft - 1 - i] = 0.5f * (here[n_cep_causal + i] + next[n_cep_causal + i]) * lifter(i, 1);
        }
        return here[split + s % harm_steps];
""")
    s = rep(s, "    auto noise_cep = [&](int64_t j, float * buf) {", "    auto noise_cep = [&](int64_t j, float * buf) -> float {")
    s = rep(s, "        buf[0] += row[n_noise_cep + s % noise_steps];\n", "        return row[n_noise_cep + s % noise_steps];\n")
    return s


# ---------------------------------------------------------------- the half-size trick with a small in-file iterative FFT instead of mtmd_audio_ifft
ITER_HELPERS = """// in-place iterative radix-2 FFT of m complex values (interleaved), exp(+2 pi i k j / m), unscaled;
// the twiddles come from the sin/cos table of the cache, which holds n = 2 m points
static void tiny_voca_cfft(const mtmd_audio_cache & cache, float * x, int m) {
    for (int i = 1, j = 0; i < m; i++) {
        int bit = m >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[2 * i], x[2 * j]);
            std::swap(x[2 * i + 1], x[2 * j + 1]);
        }
    }
    const int n = (int) cache.sin_vals.size();
    for (int len = 2; len <= m; len *= 2) {
        for (int i = 0; i < m; i += len) {
            for (int k = 0; k < len / 2; k++) {
                const float c = cache.cos_vals[k * (n / len)], s = cache.sin_vals[k * (n / len)];
                float * a = x + 2 * (i + k);
                float * b = a + len;
                const float tr = b[0] * c - b[1] * s, ti = b[0] * s + b[1] * c;
                b[0] = a[0] - tr;
                b[1] = a[1] - ti;
                a[0] += tr;
                a[1] += ti;
            }
        }
    }
}

// real FFT of n samples as a complex FFT of m = n / 2 points, the cache holds the sin/cos table of n
// split step, k = 0 .. m: z[k] = (x[k] + conj(x[m - k])) / 2 + i e^(2 pi i k / n) (x[k] - conj(x[m - k])) / 2
static void tiny_voca_split(const mtmd_audio_cache & cache, const float * x, int m, float * z) {
    for (int k = 0; k <= m; k++) {
        const float c  = cache.cos_vals[k], s = cache.sin_vals[k];
        const float er = 0.5f * (x[2 * k] + x[2 * (m - k)]),     ei = 0.5f * (x[2 * k + 1] - x[2 * (m - k) + 1]);
        const float dr = 0.5f * (x[2 * k] - x[2 * (m - k)]),     di = 0.5f * (x[2 * k + 1] + x[2 * (m - k) + 1]);
        z[2 * k]     = er - c * di - s * dr;
        z[2 * k + 1] = ei + c * dr - s * di;
    }
}

// x: n samples -> X: m + 1 bins; FFT(z) = conj(cfft(conj(z))) for z[j] = x[2 j] + i x[2 j + 1]; tmp: n + 2 floats
static void tiny_voca_rfft(const mtmd_audio_cache & cache, const float * x, int n, float * tmp, float * X) {
    const int m = n / 2;
    for (int i = 0; i < n; i += 2) {
        tmp[i]     =  x[i];
        tmp[i + 1] = -x[i + 1];
    }
    tiny_voca_cfft(cache, tmp, m);
    tmp[2 * m] = tmp[0];
    tmp[2 * m + 1] = tmp[1];
    tiny_voca_split(cache, tmp, m, X);
    for (int k = 0; k <= m; k++) {
        X[2 * k + 1] = -X[2 * k + 1];
    }
}

// X: m + 1 bins -> y: n samples, scaled by 1 / n; y: n + 2 floats
static void tiny_voca_irfft(const mtmd_audio_cache & cache, const float * X, int n, float * y) {
    tiny_voca_split(cache, X, n / 2, y);
    tiny_voca_cfft(cache, y, n / 2);
    for (int i = 0; i < n; i++) {
        y[i] *= 2.0f / n;
    }
}

"""
def iterfft(s):
    s = half(s)
    a = s.index("// real FFT of n samples as a complex FFT of m = n / 2 points with mtmd_audio_ifft")
    b = s.index("// time-varying filter: STFT")
    s = s[:a] + ITER_HELPERS + s[b:]
    s = s.replace("tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), rout.data(), fout.data());", "tiny_voca_rfft(cache, cbuf.data(), n_fft, rin.data(), fout.data());")
    s = rep(s, "tiny_voca_rfft(cache, fin.data(), n_fft, rin.data(), rout.data(), fout.data());", "tiny_voca_rfft(cache, fin.data(), n_fft, rin.data(), fout.data());")
    s = rep(s, "tiny_voca_irfft(cache, fout.data(), n_fft, rin.data(), rout.data());", "tiny_voca_irfft(cache, fout.data(), n_fft, rout.data());")
    s = rep(s, "rin(4 * n_fft), rout(4 * n_fft)", "rin(n_fft + 2), rout(n_fft + 2)")
    return s


# ---------------------------------------------------------------- iterative FFT, twiddle hoisted out of the block loop
def iter_hoist(s):
    s = iterfft(s)
    return rep(s, """    for (int len = 2; len <= m; len *= 2) {
        for (int i = 0; i < m; i += len) {
            for (int k = 0; k < len / 2; k++) {
                const float c = cache.cos_vals[k * (n / len)], s = cache.sin_vals[k * (n / len)];
                float * a = x + 2 * (i + k);
""", """    for (int len = 2; len <= m; len *= 2) {
        for (int k = 0; k < len / 2; k++) {
            const float c = cache.cos_vals[k * (n / len)], s = cache.sin_vals[k * (n / len)];
            for (int i = 0; i < m; i += len) {
                float * a = x + 2 * (i + k);
""")


IDEAS = {"J": iter_hoist, "I": iterfft, "R": resp_once, "Pc": pulse_C, "Pw": pulse_wrap, "Pt": pulse_trig, "T": tables_once, "N": norm, "H": half}

if __name__ == "__main__":
    s = ORIG
    order = ["Pc", "Pw", "Pt", "T", "N", "H", "I", "J", "R"]
    for name in sorted(sys.argv[3].split("_"), key=lambda x: order.index(x) if x in order else -1):
        if name != "ref":
            s = IDEAS[name](s)
    sys.stdout.write(s)
