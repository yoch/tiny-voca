// verbatim copy of the FFT of llama.cpp tools/mtmd/mtmd-audio.cpp (MIT, the ggml authors), own TU as in the real build
#include "shim.h"

void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n);
    cos_vals.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        sin_vals[i]  = sinf(theta);
        cos_vals[i]  = cosf(theta);
    }
}

void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i = 0; i < length; i++) {
        hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// Unified DFT implementation for both forward and inverse transforms
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued (stride 1), avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void dft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    const int sin_cos_step   = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    const float     scale = Inverse ? (1.0f / N) : 1.0f;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int   idx     = (k * n * sin_cos_step) % n_sin_cos_vals;
            float cos_val = cache.cos_vals[idx];
            float sin_val = cache.sin_vals[idx];

            if constexpr (RealInput) {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                float in_re = in[n];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                float in_re = in[n * 2 + 0];
                float in_im = in[n * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }
        }

        out[k * 2 + 0] = re * scale;
        out[k * 2 + 1] = im * scale;
    }
}

// Cooley-Tukey FFT/IFFT unified implementation
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each level
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    if (N == 1) {
        out[0] = in[0];
        if constexpr (RealInput) {
            out[1] = 0.0f;
        } else {
            out[1] = in[1];
        }
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        // Odd N: fall back to DFT
        dft_impl<Inverse, RealInput>(cache, in, N, out);
        return;
    }

    // Split into even and odd
    if constexpr (RealInput) {
        // Real input: stride is 1, copy only real values
        float * even = in + N;
        for (int i = 0; i < half_N; ++i) {
            even[i] = in[2 * i];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, true>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i] = in[2 * i + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, true>(cache, odd, half_N, odd_fft);
    } else {
        // Complex input: stride is 2, copy complex pairs
        float * even = in + N * 2;
        for (int i = 0; i < half_N; ++i) {
            even[i * 2 + 0] = in[2 * i * 2 + 0];
            even[i * 2 + 1] = in[2 * i * 2 + 1];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, false>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
            odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, false>(cache, odd, half_N, odd_fft);
    }

    float * even_fft = out + 2 * N;
    float * odd_fft  = even_fft + N;

    const int sin_cos_step = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;

    for (int k = 0; k < half_N; k++) {
        int   idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re  = cache.cos_vals[idx];
        float im  = sign * cache.sin_vals[idx];

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
        out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);

        out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
        out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
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
