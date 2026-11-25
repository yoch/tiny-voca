#ifndef TINY_VOCA_VOCODER_DSP_H
#define TINY_VOCA_VOCODER_DSP_H
#include <stddef.h>

/* ABI 1. Use aligned, zero-initialized storage of tv_context_size() bytes.
 * Initialize exclusively before sharing. Initialized contexts are immutable
 * during rendering. Calls may share inputs, but audio and scratch must each
 * be disjoint from the other buffers. Lengths count floats, except context_bytes.
 * Constants: Hann windows 512/256, then the 80/16/32 lifter values.
 * Render accepts 1..16384 acoustic frames. Failed output must be discarded.
 */
size_t tv_context_size(void);
int tv_initialize(void *context, size_t context_bytes, const float *constants, size_t count);
int tv_render(const void *context, size_t context_bytes, int frames,
              const float *pulses, size_t pulse_count, const float *harmonic, size_t harmonic_count,
              const float *noise, size_t noise_count, const float *noise_head, size_t noise_head_count,
              float *audio, size_t audio_capacity, float *scratch, size_t scratch_capacity);
/* Native parity probe for forks that compute frequency-dependent source mixes
 * before the filter. Real and imaginary arrays each have coefficients floats. */
int tv_filter(const void *context, size_t context_bytes, int branch,
              const float *source, size_t samples, const float *real, const float *imag,
              size_t coefficients, float *audio, size_t audio_capacity);
#endif
