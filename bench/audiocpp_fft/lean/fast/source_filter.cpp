#include "engine/community_models/tiny_voca/source_filter.h"

#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace engine::models::tiny_voca {
namespace {

constexpr double kPi = 3.14159265358979323846;

// 1 / (1 + n / 4) for n from `first`; coefficient 0 is the log gain and needs range.
// Filter frames transformed together.
constexpr int64_t kBlockFrames = 64;

// The transforms of a block run on the calling thread: the framework's default (0) hands
// each block of a few dozen short transforms to a thread pool, which costs more than it saves.
constexpr size_t kFftThreads = 1;

std::vector<float> lifter(int64_t size, int64_t first) {
    std::vector<float> out(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        out[static_cast<size_t>(i)] = 1.0F / (1.0F + static_cast<float>(first + i) / 4.0F);
    }
    if (first == 0 && size > 0) {
        out[0] = 2.0F;
    }
    return out;
}

// The periodic Hann window of torch.hann_window.
std::vector<float> hann(int64_t size) {
    std::vector<float> out(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        out[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(size)));
    }
    return out;
}

// exp(x) in float for the clamped log magnitudes. x = n ln 2 + r with n the integer nearest to
// x / ln 2 (the offset keeps the value positive, so the conversion rounds) and ln 2 in two parts,
// so n ln 2 is exact and |r| <= ln 2 / 2; exp(r) is its Taylor series to r^7 (the first term left
// out is below a tenth of a float ulp) and 2^n goes straight into the exponent bits. Unlike
// std::exp, a loop of it vectorizes.
inline float exp_small(float x) {
    const auto n = static_cast<std::int32_t>(x * 1.44269504F + 64.5F) - 64;
    const float r = (x - static_cast<float>(n) * 0.693145751953125F) - static_cast<float>(n) * 1.42860676533018704e-6F;
    const float p = 1.0F + r * (1.0F + r * (0.5F + r * (1.0F / 6.0F + r * (1.0F / 24.0F + r * (1.0F / 120.0F +
        r * (1.0F / 720.0F + r * (1.0F / 5040.0F)))))));
    const std::int32_t bits = (n + 127) * (1 << 23);
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// sin and cos in float for |x| <= kSincosRange. x = q pi / 2 + r with q the integer nearest to
// x 2 / pi and pi / 2 in three parts (Cephes' split, exact products for |q| < 2^13), so |r| <= pi / 4;
// Taylor series to r^9 and r^10 (first terms left out below 2e-9), then q quarter turns. Unlike
// std::sin and std::cos, a loop of it vectorizes.
constexpr float kSincosRange = 8192.0F;

inline void sincos_small(float x, float & s, float & c) {
    const auto q = static_cast<std::int32_t>(x * 0.636619772F + 8192.5F) - 8192;
    const float r = ((x - static_cast<float>(q) * 1.5703125F) - static_cast<float>(q) * 4.837512969970703125e-4F) -
                    static_cast<float>(q) * 7.54978995489188216e-8F;
    const float z = r * r;
    const float sine = r + r * z * (-1.0F / 6.0F + z * (1.0F / 120.0F + z * (-1.0F / 5040.0F + z * (1.0F / 362880.0F))));
    const float cosine = 1.0F + z * (-0.5F + z * (1.0F / 24.0F + z * (-1.0F / 720.0F + z * (1.0F / 40320.0F + z * (-1.0F / 3628800.0F)))));
    // each quarter turn maps (sin, cos) to (cos, -sin)
    const bool odd = (q & 1) != 0;
    const float s1 = odd ? cosine : sine;
    const float c1 = odd ? sine : cosine;
    s = (q & 2) != 0 ? -s1 : s1;
    c = ((q + 1) & 2) != 0 ? -c1 : c1;
}

// Rounding to the nearest integer, for |x| < 2^51: adding and taking off 1.5 * 2^52 leaves no
// fraction bits. Exact in IEEE arithmetic with the default rounding; unlike std::nearbyint and
// std::floor, the compiler vectorizes it without -fno-trapping-math. -ffast-math would fold the
// two additions away, so there the library rounding is used, which then vectorizes as well.
inline double round_small(double x) {
#ifdef __FAST_MATH__
    return std::nearbyint(x);
#else
    constexpr double kShift = 6755399441055744.0;  // 1.5 * 2^52
    return (x + kShift) - kShift;
#endif
}

inline double floor_small(double x) {
    const double r = round_small(x);
    return r > x ? r - 1.0 : r;
}

// sin and cos of 2 pi u for u >= 0 given in cycles, in double. The quarter turn nearest to u
// is taken off exactly, which leaves |x| <= pi / 4, and the Taylor series stop at x^15 and
// x^16: the first term left out is below 5e-17, under the rounding of a double. Written
// without calls or branches so that a loop of them vectorizes; libm's sincos does neither.
inline void sincos_cycles(double u, double & s, double & c) {
    const double quarter = round_small(4.0 * u);
    const double x = (u - 0.25 * quarter) * (2.0 * kPi);
    const double z = x * x;
    const double sine = x + x * z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0 + z * (1.0 / 362880.0 + z * (-1.0 / 39916800.0 +
        z * (1.0 / 6227020800.0 + z * (-1.0 / 1307674368000.0)))))));
    const double cosine = 1.0 + z * (-0.5 + z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0 + z * (-1.0 / 3628800.0 +
        z * (1.0 / 479001600.0 + z * (-1.0 / 87178291200.0 + z * (1.0 / 20922789888000.0))))))));
    // The quarter turns, (s, c) -> (c, -s) for each. Counted in doubles: mixing in integers
    // of another width keeps the loop from vectorizing.
    const double turn = quarter - 4.0 * floor_small(0.25 * quarter);
    const bool odd = turn == 1.0 || turn == 3.0;
    const double s1 = odd ? cosine : sine;
    const double c1 = odd ? sine : cosine;
    s = turn >= 2.0 ? -s1 : s1;
    c = turn == 1.0 || turn == 2.0 ? -c1 : c1;
}

}  // namespace

TinyVocaSourceFilter::TinyVocaSourceFilter(
    const TinyVocaConfig & config,
    std::vector<float> harmonic_weight,
    std::vector<float> harmonic_bias,
    std::vector<float> noise_weight,
    std::vector<float> noise_bias)
    : config_(config) {
    harmonic_.fft = config.synth_fft;
    harmonic_.hop = config.synth_hop;
    harmonic_.steps = config.hop / config.synth_hop;
    harmonic_.coefficients = config.cep_causal + config.cep_anti;
    harmonic_.lifter = lifter(config.cep_causal, 0);
    const std::vector<float> anti = lifter(config.cep_anti, 1);
    harmonic_.lifter.insert(harmonic_.lifter.end(), anti.begin(), anti.end());
    std::vector<int64_t> positions;
    for (int64_t i = 0; i < config.cep_causal; ++i) {
        positions.push_back(i);                                 // the minimum-phase part
    }
    for (int64_t i = 0; i < config.cep_anti; ++i) {
        positions.push_back(config.synth_fft - 1 - i);          // the maximum-phase part (the glottal pulse)
    }
    harmonic_.positions = positions;
    harmonic_.window = hann(harmonic_.fft);
    harmonic_.weight = std::move(harmonic_weight);
    harmonic_.bias = std::move(harmonic_bias);

    noise_.fft = config.noise_fft;
    noise_.hop = config.noise_hop;
    noise_.steps = config.hop / config.noise_hop;
    noise_.coefficients = config.noise_cep;
    noise_.lifter = lifter(config.noise_cep, 0);
    positions.clear();
    for (int64_t i = 0; i < config.noise_cep; ++i) {
        positions.push_back(i);
    }
    noise_.positions = positions;
    noise_.window = hann(noise_.fft);
    noise_.weight = std::move(noise_weight);
    noise_.bias = std::move(noise_bias);

    for (Branch * branch : {&harmonic_, &noise_}) {
        const auto outputs = static_cast<size_t>(branch->coefficients + branch->steps);
        const auto channels = static_cast<size_t>(config.channels);
        if (branch->bias.size() != outputs || branch->weight.size() != outputs * channels) {
            throw std::runtime_error("tiny-voca waveform head has an unexpected shape");
        }
        // Kept channels x outputs, so the head runs along the outputs (see shape()).
        std::vector<float> transposed(branch->weight.size());
        for (size_t o = 0; o < outputs; ++o) {
            for (size_t c = 0; c < channels; ++c) {
                transposed[c * outputs + o] = branch->weight[o * channels + c];
            }
        }
        branch->weight = std::move(transposed);
    }
}

std::vector<float> TinyVocaSourceFilter::pulse_train(const std::vector<float> & f0) const {
    // Frame values sit at samples i * hop and are interpolated in the log domain.
    // The sum of all harmonics below limit * sample_rate has the closed form
    // sin((K + 1/2) x) / (2 sin(x / 2)) - 1/2; the top harmonic fades in and out so
    // the signal stays continuous while the pitch glides. Doubles keep the phase exact.
    const auto frames = static_cast<int64_t>(f0.size());
    const int64_t hop = config_.hop;
    const auto rate = static_cast<double>(config_.sample_rate);
    const auto limit = static_cast<double>(config_.harmonic_limit);
    std::vector<double> knots(static_cast<size_t>(frames + 1));
    for (int64_t i = 0; i < frames; ++i) {
        knots[static_cast<size_t>(i)] = std::log(std::max(static_cast<double>(f0[static_cast<size_t>(i)]), 1.0));
    }
    knots[static_cast<size_t>(frames)] = frames > 0 ? knots[static_cast<size_t>(frames - 1)] : 0.0;
    std::vector<float> out(static_cast<size_t>(frames * hop));
    // Within a frame the log-linear pitch is a geometric series: one exp per frame, not per sample.
    // The phase is kept in cycles. Only the running sum of the phase steps is carried from sample
    // to sample, so that loop is a chain of additions; the reduction to [0, 1) and the waveform
    // are a loop without dependencies, which the compiler can vectorize. Below the Nyquist
    // frequency the sum within a frame stays under hop / 2 cycles, where a double still
    // resolves 1e-14 of a cycle.
    // The gain sqrt(2 / harmonics) is sqrt(frequency) times a constant below the harmonic limit:
    // a geometric series as well. (std::sqrt in the loop would not vectorize: it may set errno.)
    std::vector<double> advance(static_cast<size_t>(hop));
    std::vector<double> pitch(static_cast<size_t>(hop));
    std::vector<double> root(static_cast<size_t>(hop));
    const double gain_factor = std::sqrt(2.0 / (limit * rate));
    const double unit_gain = std::sqrt(2.0);
    double start = 0.0;
    for (int64_t i = 0; i < frames; ++i) {
        double frequency = std::exp(knots[static_cast<size_t>(i)]);
        const double ratio = std::exp((knots[static_cast<size_t>(i + 1)] - knots[static_cast<size_t>(i)]) / static_cast<double>(hop));
        double square_root = std::sqrt(frequency);
        const double root_ratio = std::sqrt(ratio);
        double sum = 0.0;
        for (int64_t m = 0; m < hop; ++m) {
            sum += frequency / rate;
            advance[static_cast<size_t>(m)] = sum;
            pitch[static_cast<size_t>(m)] = frequency;
            root[static_cast<size_t>(m)] = square_root;
            frequency *= ratio;
            square_root *= root_ratio;
        }
        float * wave_out = &out[static_cast<size_t>(i * hop)];
        for (int64_t m = 0; m < hop; ++m) {
            double cycles = start + advance[static_cast<size_t>(m)];
            cycles -= floor_small(cycles);
            const double harmonics = std::max(limit * rate / pitch[static_cast<size_t>(m)], 1.0);
            const double gain = harmonics > 1.0 ? gain_factor * root[static_cast<size_t>(m)] : unit_gain;
            const double whole = floor_small(harmonics);
            // Both angles are given in cycles, so they are reduced exactly before the sines.
            double sh = 0.0;
            double ch = 0.0;
            double st = 0.0;
            double ct = 0.0;
            sincos_cycles(0.5 * cycles, sh, ch);
            sincos_cycles((whole + 0.5) * cycles, st, ct);
            const double denominator = 2.0 * sh;
            const double core = std::fabs(denominator) < 1e-7 ? whole : st / denominator - 0.5;
            // cos((K + 1) x) from the two angles already used
            const double wave = core + (harmonics - whole) * (ct * ch - st * sh);
            wave_out[m] = static_cast<float>(wave * gain);
        }
        start += sum;
        start -= std::floor(start);
    }
    return out;
}

// Time-varying filtering of `source`: its STFT is multiplied by the response
// of each filter frame and transformed back. The result is added to `out`.
void TinyVocaSourceFilter::shape(
    const Branch & branch,
    const std::vector<float> & hidden,
    const std::vector<float> & source,
    int64_t frames,
    bool interpolate,
    std::vector<float> & out) const {
    const int64_t channels = config_.channels;
    const int64_t fft = branch.fft;
    const int64_t hop = branch.hop;
    const int64_t bins = fft / 2 + 1;
    const int64_t steps = branch.steps;
    const int64_t coefficients = branch.coefficients;
    const int64_t outputs = coefficients + steps;
    const int64_t samples = frames * config_.hop;
    // Two more filter frames cover the half window of zeros appended to the source.
    const int64_t filter_frames = steps * frames + 3;

    // The linear head: a cepstrum and one gain for each filter frame, for every acoustic frame.
    std::vector<float> head(static_cast<size_t>(frames * outputs));
    for (int64_t t = 0; t < frames; ++t) {
        const float * row = &hidden[static_cast<size_t>(t * channels)];
        // Every output adds its products in the same order as a dot product would, but the loop
        // runs along the outputs, so it vectorizes without reassociating the sums.
        float * sum = &head[static_cast<size_t>(t * outputs)];
        for (int64_t c = 0; c < channels; ++c) {
            const float * weight = &branch.weight[static_cast<size_t>(c * outputs)];
            const float x = row[c];
            for (int64_t o = 0; o < outputs; ++o) {
                sum[o] += weight[o] * x;
            }
        }
        for (int64_t o = 0; o < outputs; ++o) {
            sum[o] += branch.bias[static_cast<size_t>(o)];
        }
    }

    // The filter frames are worked a block at a time, so the memory in use does
    // not grow with the length of the utterance.
    const int64_t padded = samples + fft / 2;
    const int64_t block = std::min<int64_t>(filter_frames, kBlockFrames);
    std::vector<float> framed(static_cast<size_t>(block * fft));
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(block * bins));
    // The overlap-add of one block, from the first sample of its first frame on. The samples no
    // later frame reaches are passed to `out` after each block, so this does not grow either.
    std::vector<float> sum(static_cast<size_t>(block * hop + fft), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(block * hop + fft), 0.0F);
    // The cepstra of a block, each in a frame of `fft` samples: one batched real FFT gives the log
    // response of every filter frame, which a direct transform would take coefficients x bins for.
    std::vector<float> cepstra(static_cast<size_t>(block * fft), 0.0F);
    std::vector<std::complex<float>> response(static_cast<size_t>(block * bins));
    const auto plan = audio::get_real_fft_plan(static_cast<size_t>(fft));
    const audio::TensorStrideBytes real_strides{static_cast<std::ptrdiff_t>(fft * static_cast<int64_t>(sizeof(float))),
                                                static_cast<std::ptrdiff_t>(sizeof(float))};
    const audio::TensorStrideBytes complex_strides{static_cast<std::ptrdiff_t>(bins * static_cast<int64_t>(sizeof(std::complex<float>))),
                                                   static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    const int64_t last = frames - 1;
    for (int64_t first = 0; first < filter_frames; first += block) {
        const int64_t count = std::min(block, filter_frames - first);
        const audio::TensorShape frame_shape{static_cast<size_t>(count), static_cast<size_t>(fft)};

        // The source in windowed frames, centred, with half a window of zeros after it.
        std::fill(framed.begin(), framed.end(), 0.0F);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t start = (first + i) * hop - fft / 2;
            for (int64_t n = 0; n < fft; ++n) {
                const int64_t at = start + n;
                if (at >= 0 && at < samples) {
                    framed[static_cast<size_t>(i * fft + n)] = source[static_cast<size_t>(at)] * branch.window[static_cast<size_t>(n)];
                }
            }
        }
        plan->forward(frame_shape, real_strides, complex_strides, 1, framed.data(), spectrum.data(), 1.0F, kFftThreads);

        // Each filter frame: cepstrum -> response exp(FFT(cepstrum)), times the spectrum.
        for (int64_t i = 0; i < count; ++i) {
            const int64_t sub = std::min(first + i, steps * frames - 1);
            const int64_t here = sub / steps;
            const int64_t following = std::min((sub + 1) / steps, last);
            const float * a = &head[static_cast<size_t>(here * outputs)];
            const float * b = &head[static_cast<size_t>(following * outputs)];
            float * cepstrum = &cepstra[static_cast<size_t>(i * fft)];
            for (int64_t c = 0; c < coefficients; ++c) {
                // the harmonic envelope is interpolated between frames; the noise one is held
                const float value = interpolate ? 0.5F * (a[c] + b[c]) : a[c];
                cepstrum[branch.positions[static_cast<size_t>(c)]] = value * branch.lifter[static_cast<size_t>(c)];
            }
            cepstrum[0] += head[static_cast<size_t>(here * outputs + coefficients + sub % steps)];
        }
        plan->forward(frame_shape, real_strides, complex_strides, 1, cepstra.data(), response.data(), 1.0F, kFftThreads);
        for (int64_t i = 0; i < count; ++i) {
            std::complex<float> * row = &spectrum[static_cast<size_t>(i * bins)];
            std::complex<float> * log_response = &response[static_cast<size_t>(i * bins)];
            // The clamp gets a loop of its own: next to exp_small() it keeps the loop from vectorizing.
            // Angles this large do not come out of a trained head; they get the library's sines.
            std::int32_t outside = 0;  // a count, not a flag: || would branch
            for (int64_t k = 0; k < bins; ++k) {
                log_response[k].real(std::min(std::max(log_response[k].real(), -14.0F), 6.0F));
                outside += std::fabs(log_response[k].imag()) > kSincosRange ? 1 : 0;
            }
            if (outside > 0) {
                for (int64_t k = 0; k < bins; ++k) {
                    const float magnitude = std::exp(log_response[k].real());
                    const float angle = log_response[k].imag();
                    row[k] *= std::complex<float>(magnitude * std::cos(angle), magnitude * std::sin(angle));
                }
                continue;
            }
            for (int64_t k = 0; k < bins; ++k) {
                const float magnitude = exp_small(log_response[k].real());
                float s = 0.0F;
                float c = 0.0F;
                sincos_small(log_response[k].imag(), s, c);
                // written out: std::complex products check for infinities on every call
                const float re = magnitude * c;
                const float im = magnitude * s;
                row[k] = std::complex<float>(row[k].real() * re - row[k].imag() * im, row[k].real() * im + row[k].imag() * re);
            }
        }

        // Back to samples: overlap-add of the windowed frames over the sum of squared windows.
        plan->inverse(frame_shape, complex_strides, real_strides, 1, spectrum.data(), framed.data(), 1.0F / static_cast<float>(fft), kFftThreads);
        for (int64_t i = 0; i < count; ++i) {
            for (int64_t n = 0; n < fft; ++n) {
                const auto at = static_cast<size_t>(i * hop + n);
                const float window = branch.window[static_cast<size_t>(n)];
                sum[at] += framed[static_cast<size_t>(i * fft + n)] * window;
                envelope[at] += window * window;
            }
        }
        // Samples before the next block's first frame are complete; the rest moves to the front.
        const int64_t base = first * hop;
        const int64_t done = first + count < filter_frames ? base + count * hop : padded;
        for (int64_t at = std::max(base, fft / 2); at < done; ++at) {
            const auto j = static_cast<size_t>(at - base);
            out[static_cast<size_t>(at - fft / 2)] += envelope[j] > 1e-11F ? sum[j] / envelope[j] : sum[j];
        }
        for (std::vector<float> * buffer : {&sum, &envelope}) {
            std::copy(buffer->begin() + count * hop, buffer->end(), buffer->begin());
            std::fill(buffer->end() - count * hop, buffer->end(), 0.0F);
        }
    }
}

std::vector<float> TinyVocaSourceFilter::synthesize(
    const std::vector<float> & hidden,
    const std::vector<float> & f0,
    std::vector<float> noise,
    int64_t frames) const {
    const int64_t samples = frames * config_.hop;
    if (frames <= 0 || static_cast<int64_t>(hidden.size()) != frames * config_.channels || static_cast<int64_t>(f0.size()) != frames ||
        static_cast<int64_t>(noise.size()) < samples) {
        throw std::runtime_error("tiny-voca waveform head got inputs of the wrong size");
    }
    // The noise goes first and is let go before the pulse train is made, so
    // the two sources are never in memory together.
    std::vector<float> out(static_cast<size_t>(samples), 0.0F);
    shape(noise_, hidden, noise, frames, false, out);
    std::vector<float>().swap(noise);
    const std::vector<float> pulses = pulse_train(f0);
    shape(harmonic_, hidden, pulses, frames, true, out);
    return out;
}

}  // namespace engine::models::tiny_voca
