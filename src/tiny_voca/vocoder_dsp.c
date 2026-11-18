/* SPDX-License-Identifier: Apache-2.0
 * Real FFT source-filter DSP for Tiny Voca’s supported acoustic geometry.
 * Fixed window sizes and 4x overlap. Plans are initialized once per instance.
 */
#include <math.h>
#include "vocoder_dsp.h"
#include <stdint.h>

enum { MAX_FFT = 512, ACOUSTIC_HOP = 256, RATE = 24000 };
typedef struct {
    int n, m;
    uint16_t reverse[MAX_FFT / 2];
    float cosines[MAX_FFT / 2], sines[MAX_FFT / 2];
    float rotating_real[MAX_FFT / 2 + 1], rotating_imag[MAX_FFT / 2 + 1];
    float window[MAX_FFT], norm[MAX_FFT / 4], first_norm[MAX_FFT / 4];
} Plan;

enum { MAX_FRAMES = 16384, CONSTANTS = 896 };
typedef struct {
    Plan plans[2];
    float lifter[3][80];
    uint32_t ready;
} VocaDSP;

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
            r[j] = (unsigned)x < (unsigned)samples ? signal[x] * (window ? window[a] : 1) : 0;
            i[j] = (unsigned)(x + 1) < (unsigned)samples ? signal[x + 1] * (window ? window[a + 1] : 1) : 0;
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

static void cepstrum(const VocaDSP *ctx, const Plan *p, int branch, const float *here, const float *next,
                     Spectrum *response) {
    float coefficients[MAX_FFT] = {0};
    if (branch == 0) {
        for (int k = 0; k < 80; ++k)
            coefficients[k] = (0.5f * (here[k] + next[k])) * ctx->lifter[0][k];
        for (int k = 0; k < 16; ++k)
            coefficients[p->n - 1 - k] = (0.5f * (here[80 + k] + next[80 + k])) * ctx->lifter[1][k];
    } else {
        for (int k = 0; k < 32; ++k) coefficients[k] = here[k] * ctx->lifter[2][k];
    }
    real_fft(p, coefficients, 0, p->n, 0, response);
}

static void render_branch(const VocaDSP *ctx, int branch, int frames, const float *source, const float *head, float *out) {
    const Plan *p = &ctx->plans[branch];
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
            cepstrum(ctx, p, branch, here, head + next * width, &response);
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

/* ABI v1. The caller owns the context and the buffers. Initialize once with
 * exclusive access, then render concurrently with distinct writable buffers.
 * Lengths count floats (context_bytes counts bytes). No heap allocation.
 * Buffers must point to accessible memory of at least the declared size.
 */
size_t tv_context_size(void) { return sizeof(VocaDSP); }

static int span(const void *ptr, size_t bytes, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    return ptr && addr % alignment == 0 && bytes <= UINTPTR_MAX - addr;
}

static int disjoint(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    return aa + an <= bb || bb + bn <= aa;
}

int tv_initialize(void *storage, size_t context_bytes, const float *data, size_t count) {
    if (context_bytes < sizeof(VocaDSP) || count != CONSTANTS ||
        !span(storage, sizeof(VocaDSP), _Alignof(VocaDSP)) ||
        !span(data, CONSTANTS * sizeof(float), _Alignof(float)) ||
        !disjoint(storage, sizeof(VocaDSP), data, CONSTANTS * sizeof(float))) return 0;
    VocaDSP *ctx = storage;
    ctx->ready = 0;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(data[i]) || data[i] < 0 || data[i] > 2) return 0;
    for (int n = 256; n <= 512; n *= 2) {
        const float *window = data + (n == 256 ? 512 : 0);
        for (int i = 0; i < n / 4; ++i) {
            float weight = 0;
            for (int j = i; j < 3 * n / 4; j += n / 4) weight += window[j] * window[j];
            if (weight < 1e-6f) return 0;
        }
    }
    plan_init(&ctx->plans[0], 512, data);
    plan_init(&ctx->plans[1], 256, data + 512);
    for (int i = 0; i < 80; ++i) ctx->lifter[0][i] = data[768 + i];
    for (int i = 0; i < 16; ++i) ctx->lifter[1][i] = data[848 + i];
    for (int i = 0; i < 32; ++i) ctx->lifter[2][i] = data[864 + i];
    ctx->ready = 1;
    return 1;
}

int tv_render(const void *storage, size_t context_bytes, int frames,
              const float *pulses, size_t pulse_count, const float *harmonic, size_t harmonic_count,
              const float *noise, size_t noise_count, const float *noise_head, size_t noise_head_count,
              float *audio, size_t audio_capacity, float *scratch, size_t scratch_capacity) {
    if (frames <= 0 || frames > MAX_FRAMES || context_bytes < sizeof(VocaDSP) ||
        !span(storage, sizeof(VocaDSP), _Alignof(VocaDSP))) return 0;
    const VocaDSP *ctx = storage;
    const size_t samples = (size_t)frames * ACOUSTIC_HOP, bytes = samples * sizeof(float);
    const float *inputs[] = {pulses, harmonic, noise, noise_head};
    const size_t lengths[] = {samples, (size_t)frames * 98, samples, (size_t)frames * 36};
    const size_t counts[] = {pulse_count, harmonic_count, noise_count, noise_head_count};
    if (audio_capacity < samples || scratch_capacity < samples ||
        !span(audio, bytes, _Alignof(float)) || !span(scratch, bytes, _Alignof(float)) ||
        !disjoint(audio, bytes, scratch, bytes) ||
        !disjoint(audio, bytes, ctx, sizeof(VocaDSP)) ||
        !disjoint(scratch, bytes, ctx, sizeof(VocaDSP))) return 0;
    for (int i = 0; i < 4; ++i) {
        const size_t size = lengths[i] * sizeof(float);
        if (counts[i] != lengths[i] || !span(inputs[i], size, _Alignof(float)) ||
            !disjoint(audio, bytes, inputs[i], size) ||
            !disjoint(scratch, bytes, inputs[i], size)) return 0;
    }
    if (ctx->ready != 1) return 0;
    render_branch(ctx, 0, frames, pulses, harmonic, audio);
    render_branch(ctx, 1, frames, noise, noise_head, scratch);
    for (size_t i = 0; i < samples; ++i) audio[i] += scratch[i];
    for (size_t i = 0; i < samples; ++i)
        if (!isfinite(audio[i])) return 0;
    return 1;
}

#if !defined(__wasm__) || defined(TINY_VOCA_BENCHMARK)
/* Only frames that touch the crop contribute. The first hop has three
 * overlapping windows because the exported graph starts at frame zero. */
static void filter_spectra(const Plan *p, int samples, const float *source, const float *response_real,
            const float *response_imag, float *out) {
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

int tv_filter(const void *storage, size_t context_bytes, int branch,
              const float *source, size_t samples, const float *real, const float *imag,
              size_t coefficients, float *audio, size_t audio_capacity) {
    if (branch < 0 || branch > 1 || samples < 1 || samples > MAX_FRAMES * ACOUSTIC_HOP ||
        context_bytes < sizeof(VocaDSP) || !span(storage, sizeof(VocaDSP), _Alignof(VocaDSP))) return 0;
    const VocaDSP *ctx = storage;
    if (ctx->ready != 1) return 0;
    const Plan *p = &ctx->plans[branch];
    size_t expected = (samples / (p->n / 4) + 3) * (p->m + 1);
    size_t bytes = samples * sizeof(float), response_bytes = expected * sizeof(float);
    if (coefficients != expected || audio_capacity < samples ||
        !span(source, bytes, _Alignof(float)) || !span(audio, bytes, _Alignof(float)) ||
        !span(real, response_bytes, _Alignof(float)) || !span(imag, response_bytes, _Alignof(float)) ||
        !disjoint(audio, bytes, source, bytes) || !disjoint(audio, bytes, real, response_bytes) ||
        !disjoint(audio, bytes, imag, response_bytes) || !disjoint(audio, bytes, ctx, sizeof(VocaDSP))) return 0;
    filter_spectra(p, (int)samples, source, real, imag, audio);
    for (size_t i = 0; i < samples; ++i)
        if (!isfinite(audio[i])) return 0;
    return 1;
}
#endif
