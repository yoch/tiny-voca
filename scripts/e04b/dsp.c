/* SPDX-License-Identifier: Apache-2.0
 * Real FFT source-filter DSP for the public Tiny Voca checkpoint.
 * Fixed window sizes and 4x overlap. Plans are initialized once per instance.
 */
#include <math.h>
#include <stdint.h>

enum { MAX_FFT = 512, ACOUSTIC_HOP = 256, RATE = 24000 };
typedef struct {
    int n, m;
    uint16_t reverse[MAX_FFT / 2];
    float cosines[MAX_FFT / 2], sines[MAX_FFT / 2];
    float rotating_real[MAX_FFT / 2 + 1], rotating_imag[MAX_FFT / 2 + 1];
    float window[MAX_FFT], norm[MAX_FFT / 4], first_norm[MAX_FFT / 4];
} Plan;

static Plan plans[2];
static float lifter[3][80];

static void plan_init(Plan *p, int n, const float *window) {
    p->n = n;
    p->m = n / 2;
    const float tau = 6.2831853071795864769f;
    for (int k = 0; k < p->m; ++k) {
        int reversed = 0;
        for (int b = p->m, i = k; b > 1; b >>= 1, i >>= 1)
            reversed = 2 * reversed + (i & 1);
        p->reverse[k] = reversed;
    }
    for (int half = 1; half < p->m; half *= 2)
        for (int j = 0; j < half; ++j) {
            p->cosines[half - 1 + j] = cosf(-tau * j / (2 * half));
            p->sines[half - 1 + j] = sinf(-tau * j / (2 * half));
        }
    for (int k = 0; k <= p->m; ++k) {
        p->rotating_real[k] = cosf(-tau * k / n);
        p->rotating_imag[k] = sinf(-tau * k / n);
    }
    for (int i = 0; i < n; ++i) p->window[i] = window[i];
    for (int i = 0; i < n / 4; ++i) {
        float weight = 0;
        for (int j = i; j < 3 * n / 4; j += n / 4) weight += window[j] * window[j];
        p->first_norm[i] = 1 / weight;
        float last = window[i + 3 * n / 4];
        p->norm[i] = 1 / (weight + last * last);
    }
}

void initialize(const float *harmonic_window, const float *noise_window,
                const float *causal, const float *anti, const float *noise) {
    plan_init(&plans[0], 512, harmonic_window);
    plan_init(&plans[1], 256, noise_window);
    for (int i = 0; i < 80; ++i) lifter[0][i] = causal[i];
    for (int i = 0; i < 16; ++i) lifter[1][i] = anti[i];
    for (int i = 0; i < 32; ++i) lifter[2][i] = noise[i];
}

/* DIT butterflies on separate real/imaginary arrays. Inputs are bit-reversed.
 * The inner loop uses contiguous lanes, for native and wasm auto-vectorization.
 */
static void butterflies(const Plan *p, float *restrict r, float *restrict i) {
    for (int half = 1; half < p->m; half *= 2) {
        const float *cosine = p->cosines + half - 1;
        const float *sine = p->sines + half - 1;
        for (int block = 0; block < p->m; block += 2 * half)
            for (int j = 0; j < half; ++j) {
                int a = block + j, b = a + half;
                float tr = r[b] * cosine[j] - i[b] * sine[j];
                float ti = r[b] * sine[j] + i[b] * cosine[j];
                r[b] = r[a] - tr;
                i[b] = i[a] - ti;
                r[a] += tr;
                i[a] += ti;
            }
    }
}

typedef struct { float r[MAX_FFT / 2 + 1], i[MAX_FFT / 2 + 1]; } Spectrum;

static void real_fft(const Plan *p, const float *signal, int start, int samples,
                     const float *window, Spectrum *out) {
    float r[MAX_FFT / 2], i[MAX_FFT / 2];
    if (start >= 0 && start + p->n <= samples) {
        for (int j = 0; j < p->m; ++j) {
            int a = 2 * p->reverse[j];
            r[j] = signal[start + a] * (window ? window[a] : 1);
            i[j] = signal[start + a + 1] * (window ? window[a + 1] : 1);
        }
    } else {
        for (int j = 0; j < p->m; ++j) {
            int a = 2 * p->reverse[j], x = start + a;
            r[j] = (unsigned)x < (unsigned)samples ? signal[x] * window[a] : 0;
            i[j] = (unsigned)(x + 1) < (unsigned)samples ? signal[x + 1] * window[a + 1] : 0;
        }
    }
    butterflies(p, r, i);
    out->r[0] = r[0] + i[0];
    out->r[p->m] = r[0] - i[0];
    out->i[0] = out->i[p->m] = 0;
    for (int k = 1; k < p->m; ++k) {
        float dr = r[k] - r[p->m - k], di = i[k] + i[p->m - k];
        out->r[k] = 0.5f * (r[k] + r[p->m - k] + di * p->rotating_real[k] + dr * p->rotating_imag[k]);
        out->i[k] = 0.5f * (i[k] - i[p->m - k] - dr * p->rotating_real[k] + di * p->rotating_imag[k]);
    }
}

static void inverse_add(const Plan *p, const Spectrum *x, int start, int samples, float *out) {
    float r[MAX_FFT / 2], i[MAX_FFT / 2];
    for (int j = 0; j < p->m; ++j) {
        int k = p->reverse[j], other = p->m - k;
        float dr = x->r[k] - x->r[other], di = x->i[k] + x->i[other];
        /* Conjugate the packed inverse, so it uses the same butterflies. */
        r[j] = 0.5f * (x->r[k] + x->r[other] - di * p->rotating_real[k] + dr * p->rotating_imag[k]);
        i[j] = -0.5f * (x->i[k] - x->i[other] + dr * p->rotating_real[k] + di * p->rotating_imag[k]);
    }
    butterflies(p, r, i);
    const float scale = 1.0f / p->m;
    if (start >= 0 && start + p->n <= samples) {
        for (int j = 0; j < p->m; ++j) {
            out[start + 2*j] += (r[j] * scale) * p->window[2*j];
            out[start + 2*j + 1] -= (i[j] * scale) * p->window[2*j + 1];
        }
    } else {
        for (int j = 0; j < p->m; ++j) {
            int a = 2 * j, dst = start + a;
            if ((unsigned)dst < (unsigned)samples) out[dst] += (r[j] * scale) * p->window[a];
            if ((unsigned)(dst + 1) < (unsigned)samples) out[dst + 1] -= (i[j] * scale) * p->window[a + 1];
        }
    }
}

static void multiply(const Plan *p, Spectrum *x, const float *real, const float *imag) {
    for (int k = 0; k <= p->m; ++k) {
        float r = x->r[k], i = x->i[k];
        x->r[k] = r * real[k] - i * imag[k];
        x->i[k] = r * imag[k] + i * real[k];
    }
    x->i[0] = x->i[p->m] = 0;
}

static void finish(const Plan *p, int samples, float *out) {
    const int hop = p->n / 4;
    for (int i = 0; i < hop && i < samples; ++i) out[i] *= p->first_norm[i];
    if (samples <= hop) return;
    const int end = samples / hop * hop;
    for (int offset = hop; offset < end; offset += hop)
        for (int j = 0; j < hop; ++j) out[offset + j] *= p->norm[j];
    for (int j = 0; j < samples - end; ++j) out[end + j] *= p->norm[j];
}

/* Only frames that touch the crop contribute. The first hop has three
 * overlapping windows because the exported graph starts at frame zero. */
void filter(int branch, int samples, const float *source, const float *response_real,
            const float *response_imag, float *out) {
    const Plan *p = &plans[branch];
    const int hop = p->n / 4;
    for (int k = 0; k < samples; ++k) out[k] = 0;
    for (int frame = 0; frame * hop < samples + p->n / 2; ++frame) {
        int start = frame * hop - p->n / 2;
        Spectrum spectrum;
        real_fft(p, source, start, samples, p->window, &spectrum);
        multiply(p, &spectrum, response_real + frame * (p->m + 1), response_imag + frame * (p->m + 1));
        inverse_add(p, &spectrum, start, samples, out);
    }
    finish(p, samples, out);
}

static void cepstrum(const Plan *p, int branch, const float *here, const float *next,
                     Spectrum *response) {
    float coefficients[MAX_FFT] = {0};
    if (branch == 0) {
        for (int k = 0; k < 80; ++k)
            coefficients[k] = (0.5f * (here[k] + next[k])) * lifter[0][k];
        for (int k = 0; k < 16; ++k)
            coefficients[p->n - 1 - k] = (0.5f * (here[80 + k] + next[80 + k])) * lifter[1][k];
    } else {
        for (int k = 0; k < 32; ++k) coefficients[k] = here[k] * lifter[2][k];
    }
    real_fft(p, coefficients, 0, p->n, 0, response);
}

void compact_filter(int branch, int frames, const float *source, const float *head, float *out) {
    const Plan *p = &plans[branch];
    const int samples = frames * ACOUSTIC_HOP, hop = p->n / 4;
    const int steps = ACOUSTIC_HOP / hop, width = branch ? 36 : 98;
    const int split = width - steps;
    Spectrum response, phase;
    for (int k = 0; k < samples; ++k) out[k] = 0;
    for (int frame = 0; frame * hop < samples + p->n / 2; ++frame) {
        const int sub = frame < frames * steps ? frame : frames * steps - 1;
        const int t = sub / steps;
        const float *here = head + t * width;
        if (frame < frames * steps && (branch == 0 || frame % steps == 0)) {
            int next = (sub + 1) / steps;
            if (next >= frames) next = frames - 1;
            cepstrum(p, branch, here, head + next * width, &response);
            for (int k = 0; k <= p->m; ++k) {
                phase.r[k] = cosf(response.i[k]);
                phase.i[k] = sinf(response.i[k]);
            }
        }
        const float gain = here[split + sub % steps];
        Spectrum spectrum, transfer;
        real_fft(p, source, frame * hop - p->n / 2, samples, p->window, &spectrum);
        for (int k = 0; k <= p->m; ++k) {
            float log_magnitude = response.r[k] + gain;
            if (log_magnitude > 6) log_magnitude = 6;
            if (log_magnitude < -14) log_magnitude = -14;
            float magnitude = expf(log_magnitude);
            transfer.r[k] = magnitude * phase.r[k];
            transfer.i[k] = magnitude * phase.i[k];
        }
        multiply(p, &spectrum, transfer.r, transfer.i);
        inverse_add(p, &spectrum, frame * hop - p->n / 2, samples, out);
    }
    finish(p, samples, out);
}

/* Same phase scheme as Exportable._pulses. Transcendentals are supplied by the
 * target libm, so numerical parity with ORT needs independent verification.
 */
void oscillator(int frames, const float *f0, float *out) {
    unsigned phase = 0;
    const unsigned phase_steps = 1u << 24;
    const float tau = 6.2831853071795864769f;
    float low = logf(f0[0]);
    for (int t = 0; t < frames; ++t) {
        float high = t + 1 < frames ? logf(f0[t + 1]) : low;
        float within = 0, start = (float)phase / phase_steps;
        for (int j = 0; j < ACOUSTIC_HOP; ++j) {
            float weight = (float)j / ACOUSTIC_HOP;
            float frequency = expf(low * (1 - weight) + high * weight);
            within += frequency / RATE;
            float cycles = within + start;
            float theta = tau * (cycles - floorf(cycles));
            float harmonics = (0.47f * RATE) / frequency;
            float whole = floorf(harmonics);
            float denominator = 2 * sinf(theta / 2);
            float core = fabsf(denominator) < 1e-6f ? whole : sinf((whole + 0.5f) * theta) / denominator - 0.5f;
            out[t * ACOUSTIC_HOP + j] = (core + (harmonics - whole) * cosf((whole + 1) * theta)) * sqrtf(2 / harmonics);
        }
        phase = (phase + (unsigned)nearbyintf((within - floorf(within)) * phase_steps)) & (phase_steps - 1);
        low = high;
    }
}

void mix(int samples, float *restrict a, const float *restrict b) {
    for (int i = 0; i < samples; ++i) a[i] += b[i];
}
