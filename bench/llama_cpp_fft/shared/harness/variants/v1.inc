// Unified DFT implementation for both forward and inverse transforms, used for the odd part of N
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued, avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag)
// the input is read with a stride of stride values (real or complex)
template <bool Inverse, bool RealInput>
static void dft_impl(const mtmd_audio_cache & cache, const float * in, int N, int stride, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    const int sin_cos_step   = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    const float     scale = Inverse ? (1.0f / N) : 1.0f;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        int idx = 0; // (k * n * sin_cos_step) % n_sin_cos_vals
        for (int n = 0; n < N; n++) {
            float cos_val = cache.cos_vals[idx];
            float sin_val = cache.sin_vals[idx];

            if constexpr (RealInput) {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                float in_re = in[n * stride];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                float in_re = in[n * stride * 2 + 0];
                float in_im = in[n * stride * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }

            idx += k * sin_cos_step;
            if (idx >= n_sin_cos_vals) {
                idx -= n_sin_cos_vals;
            }
        }

        out[k * 2 + 0] = re * scale;
        out[k * 2 + 1] = im * scale;
    }
}

// Cooley-Tukey FFT/IFFT unified implementation, iterative: N = 2^p * M with M odd
// the 2^p subsequences in[r + 2^p * n] are transformed with the DFT (r in bit-reversed order), then
// p radix-2 stages combine them in place in out
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each stage
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
// in is not modified, out needs 2 * N floats
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    int n_odd = N;
    int n_bits = 0;
    while (n_odd % 2 == 0) {
        n_odd /= 2;
        n_bits++;
    }
    const int n_sub = N / n_odd;

    for (int j = 0; j < n_sub; j++) {
        int r = 0;
        for (int b = 0; b < n_bits; b++) {
            r |= ((j >> b) & 1) << (n_bits - 1 - b);
        }
        float * dst = out + 2 * j * n_odd;
        if (n_odd > 1) {
            dft_impl<Inverse, RealInput>(cache, in + (RealInput ? r : 2 * r), n_odd, n_sub, dst);
        } else if constexpr (RealInput) {
            dst[0] = in[r];
            dst[1] = 0.0f;
        } else {
            dst[0] = in[2 * r + 0];
            dst[1] = in[2 * r + 1];
        }
    }

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;

    for (int len = 2 * n_odd; len <= N; len *= 2) {
        const int half_N       = len / 2;
        const int sin_cos_step = n_sin_cos_vals / len;
        for (int i = 0; i < N; i += len) {
            float * even_fft = out + 2 * i;
            float * odd_fft  = even_fft + 2 * half_N;
            for (int k = 0; k < half_N; k++) {
                int   idx = k * sin_cos_step;  // t = 2*M_PI*k/len
                float re  = cache.cos_vals[idx];
                float im  = sign * cache.sin_vals[idx];

                float re_even = even_fft[2 * k + 0];
                float im_even = even_fft[2 * k + 1];
                float re_odd  = odd_fft[2 * k + 0];
                float im_odd  = odd_fft[2 * k + 1];

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
