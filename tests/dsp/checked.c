#include "vocoder_dsp.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void *allocate(size_t count, size_t width) {
    void *block = calloc(count, width);
    assert(block);
    return block;
}

static void window(float *out, int length) {
    for (int i = 0; i < length; ++i)
        out[i] = 0.5f - 0.5f * cosf(6.283185307179586f * i / length);
}

static uint32_t step(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void exercise(const void *context, int frames, uint32_t seed) {
    size_t s = (size_t)frames * 256, h = (size_t)frames * 98, n = (size_t)frames * 36;
    size_t context_bytes = tv_context_size();
    float *pulses = allocate(s, 4), *harmonic = allocate(h, 4), *noise = allocate(s, 4);
    float *noise_head = allocate(n, 4), *audio = allocate(s + 1, 4), *scratch = allocate(s, 4);
    for (size_t i = 0; i < s; ++i) {
        pulses[i] = sinf(0.12f * (float)i + 0.5f * sinf((float)i / 150));
        noise[i] = ((float)(step(&seed) >> 8) / 8388608) - 1;
    }
    audio[s] = 123;
    assert(tv_render(context, context_bytes, frames,
                     pulses, s, harmonic, h, noise, s, noise_head, n, audio, s + 1, scratch, s));
    for (size_t i = 0; i < s; ++i)
        assert(fabsf(audio[i] - (pulses[i] + noise[i])) < 3e-6f);
    assert(audio[s] == 123);
    if (frames <= 8) {
        const size_t coefficients = (s / 128 + 3) * 257;
        float *real = allocate(coefficients,4), *imag = allocate(coefficients,4);
        for (size_t i = 0; i < coefficients; ++i) real[i] = 1;
        assert(tv_filter(context,context_bytes,0,pulses,s,real,imag,coefficients,audio,s));
        for (size_t i = 0; i < s; ++i) assert(fabsf(audio[i] - pulses[i]) < 3e-6f);
        assert(!tv_filter(context,context_bytes,0,pulses,s,real,imag,coefficients-1,audio,s));
        assert(!tv_filter(context,context_bytes,2,pulses,s,real,imag,coefficients,audio,s));
        assert(!tv_filter(context,context_bytes,0,pulses,s,real,imag,coefficients,pulses,s));
        free(imag);
        free(real);
    }
    for (int i = 0; i < 64; ++i) {
        unsigned fault = step(&seed) >> 29;
        int bad_frames = fault == 0 ? INT_MIN : fault == 1 ? INT_MAX : frames;
        assert(!tv_render(context, context_bytes - (fault == 7), bad_frames,
                          pulses, s - (fault == 2), harmonic, h - (fault == 3),
                          noise, s, noise_head, n - (fault == 4), audio, s - (fault == 5),
                          fault == 6 ? audio + 1 : scratch, s));
        assert(audio[s] == 123);
    }
    harmonic[0] = NAN;
    assert(!tv_render(context, context_bytes, frames,
                      pulses, s, harmonic, h, noise, s, noise_head, n, audio, s, scratch, s));
    free(scratch);
    free(audio);
    free(noise_head);
    free(noise);
    free(harmonic);
    free(pulses);
}

static void *parallel(void *context) {
    for (int i = 1; i <= 12; ++i) exercise(context, i, (uint32_t)i);
    return NULL;
}

int main(void) {
    float constants[896] = {0};
    void *first = allocate(tv_context_size(), 1), *second = allocate(tv_context_size(), 1);
    assert(!tv_initialize(first, tv_context_size(), constants, 896));
    window(constants, 512);
    window(constants + 512, 256);
    for (int i = 768; i < 896; ++i) constants[i] = 1;
    assert(!tv_initialize(first, tv_context_size() - 1, constants, 896));
    assert(!tv_initialize(first, tv_context_size(), constants, 895));
    assert(tv_initialize(first, tv_context_size(), constants, 896));
    constants[0] = 0.25f;
    assert(tv_initialize(second, tv_context_size(), constants, 896));
    pthread_t workers[3];
    assert(pthread_create(&workers[0], NULL, parallel, first) == 0);
    assert(pthread_create(&workers[1], NULL, parallel, first) == 0);
    assert(pthread_create(&workers[2], NULL, parallel, second) == 0);
    uint32_t rng = 32401;
    for (int i = 0; i < 64; ++i) {
        unsigned frames = (step(&rng) >> 24) % 80 + 1;
        exercise(i % 2 ? first : second, (int)frames, rng);
    }
    exercise(first, 512, 9);
    exercise(first, 3000, 8);
    for (int i = 0; i < 3; ++i) assert(pthread_join(workers[i], NULL) == 0);
    free(second);
    free(first);
    puts("DSP capacity fuzz, identity, repeated long calls and shared-context concurrency passed");
}
