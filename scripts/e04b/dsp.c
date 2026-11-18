/* E04-B benchmark adapter. Only the benchmark uses this process-wide context.
 * The pilot exports the checked, context-based ABI from the shared core. */
#define TINY_VOCA_BENCHMARK 1
#include "../../src/tiny_voca/vocoder_dsp.c"

static VocaDSP benchmark;
void initialize(const float *harmonic_window, const float *noise_window,
                const float *causal, const float *anti, const float *noise) {
    plan_init(&benchmark.plans[0], 512, harmonic_window);
    plan_init(&benchmark.plans[1], 256, noise_window);
    for (int i = 0; i < 80; ++i) benchmark.lifter[0][i] = causal[i];
    for (int i = 0; i < 16; ++i) benchmark.lifter[1][i] = anti[i];
    for (int i = 0; i < 32; ++i) benchmark.lifter[2][i] = noise[i];
}

void filter(int branch, int samples, const float *source, const float *response_real,
            const float *response_imag, float *out) {
    filter_spectra(&benchmark.plans[branch], samples, source, response_real, response_imag, out);
}

void compact_filter(int branch, int frames, const float *source, const float *head, float *out) {
    render_branch(&benchmark, branch, frames, source, head, out);
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
