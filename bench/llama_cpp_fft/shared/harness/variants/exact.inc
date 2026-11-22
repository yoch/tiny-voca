// FFT/IFFT of N = n_sub * n_odd points, n_sub a power of two and n_odd odd (any N > 0):
//   1. DFT of size n_odd of each of the n_sub subsequences in[r + n_sub * n], all r at once,
//      the result of subsequence r goes to block bitrev(r) of out
//   2. in-place radix-2 stages on blocks of 2 * n_odd, 4 * n_odd, ..., N values; for n_odd == 1 the first stage
//      (twiddle 1) is done while gathering the input in bit-reversed order
// each output value goes through the same operations as in the recursive Cooley-Tukey this replaces
// Template parameters:
//   Inverse: false = FFT with exp(-2 pi i k / N), no scaling
//            true  = IFFT with exp(+2 pi i k / N), scales by 1/N (1/n_odd in the DFT, 0.5 per stage)
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
// in is not modified, out needs 2 * N floats + 2 * max(N / 2, n_sub) floats of scratch
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;
    constexpr int   n_in  = RealInput ? 1 : 2; // floats per input value

    int n_odd = N;
    while (n_odd % 2 == 0) {
        n_odd /= 2;
    }
    const int n_sub = N / n_odd;

    // j = bitrev(r) for r = 0, 1, ...: add 1 to j starting from its highest bit h
    auto bitrev_next = [](int & j, int h) {
        for (; j & h; h >>= 1) {
            j ^= h;
        }
        j |= h;
    };

    float * scratch = out + 2 * N;
    int     len     = 2 * n_odd; // size of the blocks after the next stage
    if (n_odd > 1) {
        const int   sin_cos_step = n_sin_cos_vals / n_odd;
        const float dft_scale    = Inverse ? (1.0f / n_odd) : 1.0f;
        float * acc = scratch;
        for (int k = 0; k < n_odd; k++) {
            std::fill(acc, acc + 2 * n_sub, 0.0f);
            int idx = 0; // (k * n * sin_cos_step) % n_sin_cos_vals
            for (int n = 0; n < n_odd; n++) {
                const float   cos_val = cache.cos_vals[idx];
                const float   sin_val = cache.sin_vals[idx];
                const float * x       = in + n * n_sub * n_in;
                for (int r = 0; r < n_sub; r++) {
                    if constexpr (RealInput) {
                        acc[2 * r + 0] += x[r] * cos_val;
                        acc[2 * r + 1] += sign * x[r] * sin_val;
                    } else {
                        // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                        acc[2 * r + 0] += x[2 * r + 0] * cos_val - sign * x[2 * r + 1] * sin_val;
                        acc[2 * r + 1] += sign * x[2 * r + 0] * sin_val + x[2 * r + 1] * cos_val;
                    }
                }
                idx += k * sin_cos_step;
                if (idx >= n_sin_cos_vals) {
                    idx -= n_sin_cos_vals;
                }
            }
            for (int r = 0, j = 0; r < n_sub; r++, bitrev_next(j, n_sub / 2)) {
                out[2 * (j * n_odd + k) + 0] = acc[2 * r + 0] * dft_scale;
                out[2 * (j * n_odd + k) + 1] = acc[2 * r + 1] * dft_scale;
            }
        }
    } else if (N == 1) {
        out[0] = in[0];
        out[1] = RealInput ? 0.0f : in[1];
        return;
    } else {
        // size 1 subsequences: the values r and r + N / 2 form block bitrev(r) of the first stage
        const int half_N = N / 2;
        for (int j = 0, r = 0; j < half_N; j++, bitrev_next(r, half_N / 2)) {
            const float * a = in + r * n_in;
            const float * b = in + (r + half_N) * n_in;
            if constexpr (RealInput) {
                out[4 * j + 0] = a[0] + b[0];
                out[4 * j + 1] = 0.0f;
                out[4 * j + 2] = a[0] - b[0];
                out[4 * j + 3] = 0.0f;
            } else {
                out[4 * j + 0] = scale * (a[0] + b[0]);
                out[4 * j + 1] = scale * (a[1] + b[1]);
                out[4 * j + 2] = scale * (a[0] - b[0]);
                out[4 * j + 3] = scale * (a[1] - b[1]);
            }
        }
        len = 4;
    }

    for (; len <= N; len *= 2) {
        const int half_N       = len / 2;
        const int sin_cos_step = n_sin_cos_vals / len;
        // twiddles of this stage, contiguous
        float * twiddle = scratch;
        for (int k = 0; k < half_N; k++) {
            twiddle[2 * k + 0] = cache.cos_vals[k * sin_cos_step];  // t = 2*M_PI*k/len
            twiddle[2 * k + 1] = sign * cache.sin_vals[k * sin_cos_step];
        }
        for (int i = 0; i < N; i += len) {
            float * even_fft = out + 2 * i;
            float * odd_fft  = even_fft + 2 * half_N;
            for (int k = 0; k < half_N; k++) {
                const float re      = twiddle[2 * k + 0];
                const float im      = twiddle[2 * k + 1];
                const float re_even = even_fft[2 * k + 0];
                const float im_even = even_fft[2 * k + 1];
                const float re_odd  = odd_fft[2 * k + 0];
                const float im_odd  = odd_fft[2 * k + 1];

                even_fft[2 * k + 0] = scale * (re_even + re * re_odd - im * im_odd);
                even_fft[2 * k + 1] = scale * (im_even + re * im_odd + im * re_odd);

                odd_fft[2 * k + 0] = scale * (re_even - re * re_odd + im * im_odd);
                odd_fft[2 * k + 1] = scale * (im_even - re * im_odd - im * re_odd);
            }
        }
    }
}

// Forward FFT for real input (used by mel spectrogram)
static void fft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<false, true>(cache, in, N, out);
}

// Inverse FFT for complex input
static void ifft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<true, false>(cache, in, N, out);
}

void mtmd_audio_fft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft(cache, in, N, out);
}

void mtmd_audio_ifft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    ifft(cache, in, N, out);
}
