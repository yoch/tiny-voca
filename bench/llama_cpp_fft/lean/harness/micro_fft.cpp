// FFT microbenchmark (self-contained: the FFTs of src/I.cpp, src/J.cpp and src/C.cpp pasted in): mtmd_audio_fft/ifft vs small iterative FFTs vs variant C
// build: g++ -O3 -DNDEBUG -fPIC -std=c++17 -Iharness harness/micro_fft.cpp harness/fft.cpp
#include "shim.h"
#include <chrono>
namespace I {
// in-place iterative radix-2 FFT of m complex values (interleaved), exp(+2 pi i k j / m), unscaled;
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

}
namespace J {
// in-place iterative radix-2 FFT of m complex values (interleaved), exp(+2 pi i k j / m), unscaled;
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
        for (int k = 0; k < len / 2; k++) {
            const float c = cache.cos_vals[k * (n / len)], s = cache.sin_vals[k * (n / len)];
            for (int i = 0; i < m; i += len) {
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



namespace K {
// split arrays, stage loop like C, twiddles from the cache table with stride n / len
static void cfft(const mtmd_audio_cache & cache, float * r, float * im, int m) {
    const int n = (int) cache.sin_vals.size();
    for (int half = 1; half < m; half *= 2) {
        const int step = n / (2 * half);
        for (int block = 0; block < m; block += 2 * half) {
            for (int j = 0; j < half; j++) {
                const float c = cache.cos_vals[j * step], s = cache.sin_vals[j * step];
                const int a = block + j, b = a + half;
                const float tr = r[b] * c - im[b] * s, ti = r[b] * s + im[b] * c;
                r[b] = r[a] - tr; im[b] = im[a] - ti;
                r[a] += tr; im[a] += ti;
            }
        }
    }
}
// C's generic loop from half = 1 with contiguous twiddles (no fused stages)
static void cnofuse(const tiny_voca_rfft & p, float * r, float * im) {
    const int m = p.m;
    for (int half = 1; half < m; half *= 2) {
        const float * c = p.tw_c.data() + half - 1;
        const float * s = p.tw_s.data() + half - 1;
        for (int block = 0; block < m; block += 2 * half) {
            for (int j = 0; j < half; j++) {
                const int a = block + j, b = a + half;
                const float tr = r[b] * c[j] - im[b] * s[j], ti = r[b] * s[j] + im[b] * c[j];
                r[b] = r[a] - tr; im[b] = im[a] - ti;
                r[a] += tr; im[a] += ti;
            }
        }
    }
}
// interleaved, contiguous per-call twiddle array built from the table (stack), J loop order
static void cfft_il(const mtmd_audio_cache & cache, float * x, int m) {
    const int n = (int) cache.sin_vals.size();
    for (int len = 2; len <= m; len *= 2) {
        for (int i = 0; i < m; i += len) {
            float * a = x + 2 * i;
            float * b = a + len;
            for (int k = 0; k < len / 2; k++) {
                const float c = cache.cos_vals[k * (n / len)], s = cache.sin_vals[k * (n / len)];
                const float tr = b[2*k] * c - b[2*k+1] * s, ti = b[2*k] * s + b[2*k+1] * c;
                b[2*k] = a[2*k] - tr; b[2*k+1] = a[2*k+1] - ti;
                a[2*k] += tr; a[2*k+1] += ti;
            }
        }
    }
}
}
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    for (int n : {256, 512}) {
        int m = n / 2;
        mtmd_audio_cache cache; cache.fill_sin_cos_table(n);
        std::vector<float> x(16 * n), y(16 * n), r(m), im(m), X(n + 2), Y(n + 2);
        for (int i = 0; i < 2 * m; i++) x[i] = (i * 7919 % 113) / 113.0f;
        tiny_voca_rfft p(n);
        const int R = 20000;
        double t0 = now(); for (int i = 0; i < R; i++) mtmd_audio_ifft(cache, x.data(), m, y.data()); double t1 = now();
        for (int i = 0; i < R; i++) I::tiny_voca_cfft(cache, x.data(), m); double t2 = now();
        for (int i = 0; i < R; i++) J::tiny_voca_cfft(cache, x.data(), m); double t3 = now();
        for (int i = 0; i < R; i++) p.butterflies(r.data(), im.data()); double t4 = now();
        for (int i = 0; i < R; i++) mtmd_audio_fft(cache, x.data(), n, y.data()); double t5 = now();
        for (int i = 0; i < R; i++) p.forward(x.data(), X.data(), Y.data()); double t6 = now();
        double u0 = now(); for (int i = 0; i < R; i++) K::cfft(cache, r.data(), im.data(), m); double u1 = now();
        for (int i = 0; i < R; i++) K::cnofuse(p, r.data(), im.data()); double u2 = now();
        for (int i = 0; i < R; i++) K::cfft_il(cache, x.data(), m); double u3 = now();
        printf("   K split+stride-tw %.2f | C no-fuse %.2f | interleaved no-bitrev %.2f\n", (u1-u0)/R, (u2-u1)/R, (u3-u2)/R);
        printf("n=%d (complex m=%d): mtmd_ifft(m) %.2f us | I %.2f | J %.2f | C butterflies %.2f | mtmd_fft(n real) %.2f | C forward(n) %.2f\n", n, m,
            (t1-t0)/R, (t2-t1)/R, (t3-t2)/R, (t4-t3)/R, (t5-t4)/R, (t6-t5)/R);
    }
}
