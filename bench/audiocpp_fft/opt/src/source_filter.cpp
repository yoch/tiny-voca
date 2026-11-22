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

// Filter frames transformed together.
constexpr int64_t kBlockFrames = 64;

// Acoustic frames worked together by the linear head.
constexpr int64_t kHeadFrames = 32;

// The transforms of a block run on the calling thread: the framework's default (0) hands
// each block of a few dozen short transforms to a thread pool, which costs more than it saves.
constexpr size_t kFftThreads = 1;

// exp_bounded() holds its accuracy up to here; the noise branch uses exp(a) exp(gain) within it.
constexpr float kLevelRange = 40.0F;

// 1 / (1 + n / 4) for n from `first`; coefficient 0 is the log gain and needs range.
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

// Rounding to the nearest integer, for |x| < 2^51 (double) or 2^22 (float): adding and taking
// off 1.5 * 2^52 (or 1.5 * 2^23) leaves no fraction bits. Exact in IEEE arithmetic with the
// default rounding; unlike std::nearbyint and std::floor, the compiler vectorizes it without
// -fno-trapping-math. -ffast-math would fold the two additions away, so there the library
// rounding is used, which then vectorizes as well.
inline double round_small(double x) {
#ifdef __FAST_MATH__
    return std::nearbyint(x);
#else
    constexpr double kShift = 6755399441055744.0;  // 1.5 * 2^52
    return (x + kShift) - kShift;
#endif
}

inline float round_small(float x) {
#ifdef __FAST_MATH__
    return std::nearbyint(x);
#else
    constexpr float kShift = 12582912.0F;  // 1.5 * 2^23
    return (x + kShift) - kShift;
#endif
}

inline double floor_small(double x) {
    const double r = round_small(x);
    return r > x ? r - 1.0 : r;
}

inline float floor_small(float x) {
    const float r = round_small(x);
    return r > x ? r - 1.0F : r;
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

// exp(x) in float for |x| <= 40, which holds the clamped log magnitudes. x = n ln 2 + r with
// |r| <= ln 2 / 2 (ln 2 in two parts, so n ln 2 is exact), and the Taylor series of exp(r) to
// r^7: the first term left out is below 6e-9, a tenth of a float ulp; measured error 1.7 ulp.
// 2^n is written straight into the exponent bits. No call and no branch, so a loop of it
// vectorizes.
inline float exp_bounded(float x) {
    const float n = round_small(x * 1.44269504088896341F);
    const float r = (x - n * 0.693145751953125F) - n * 1.42860676533018704e-6F;
    const float p = 1.0F + r * (1.0F + r * (0.5F + r * (1.0F / 6.0F + r * (1.0F / 24.0F + r * (1.0F / 120.0F +
        r * (1.0F / 720.0F + r * (1.0F / 5040.0F)))))));
    const std::int32_t bits = (static_cast<std::int32_t>(n) + 127) << 23;
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// sin and cos in float for |x| <= kSincosRange. The nearest multiple q of pi / 2 is taken off
// with pi / 2 in three parts (the first two products are exact for |q| < 2^13, Cephes'
// split), which leaves |r| <= pi / 4; the Taylor series stop at r^9 and r^10, whose first
// terms left out are below 2e-9. No call and no branch, so a loop of it vectorizes.
constexpr float kSincosRange = 8192.0F;

inline void sincos_bounded(float x, float & s, float & c) {
    const float q = round_small(x * 0.636619772367581343F);
    const float r = ((x - q * 1.5703125F) - q * 4.837512969970703125e-4F) - q * 7.54978995489188216e-8F;
    const float z = r * r;
    const float sine = r + r * z * (-1.0F / 6.0F + z * (1.0F / 120.0F + z * (-1.0F / 5040.0F + z * (1.0F / 362880.0F))));
    const float cosine = 1.0F + z * (-0.5F + z * (1.0F / 24.0F + z * (-1.0F / 720.0F + z * (1.0F / 40320.0F + z * (-1.0F / 3628800.0F)))));
    // the quarter turns, (s, c) -> (c, -s) for each; q is a float, as integers of another
    // width than the data keep the loop from vectorizing
    const float turn = q - 4.0F * floor_small(0.25F * q);
    const bool odd = turn == 1.0F || turn == 3.0F;
    const float s1 = odd ? cosine : sine;
    const float c1 = odd ? sine : cosine;
    s = turn >= 2.0F ? -s1 : s1;
    c = turn == 1.0F || turn == 2.0F ? -c1 : c1;
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
    // Each output is a dot product added up in channel order, as in the reference, but for
    // kHeadFrames frames at once: their states are transposed to channels x frames, so the
    // innermost loop runs along the frames, vectorizes, and its sums stay in registers while
    // every weight is read once per tile.
    std::vector<float> head(static_cast<size_t>(frames * outputs));
    std::vector<float> states(static_cast<size_t>(channels * kHeadFrames));
    for (int64_t t0 = 0; t0 < frames; t0 += kHeadFrames) {
        const int64_t tile = std::min(kHeadFrames, frames - t0);
        for (int64_t c = 0; c < channels; ++c) {
            float * column = &states[static_cast<size_t>(c * kHeadFrames)];
            for (int64_t t = 0; t < tile; ++t) {
                column[t] = hidden[static_cast<size_t>((t0 + t) * channels + c)];
            }
            std::fill(column + tile, column + kHeadFrames, 0.0F);
        }
        for (int64_t o = 0; o < outputs; ++o) {
            const float * weight = &branch.weight[static_cast<size_t>(o * channels)];
            float sums[kHeadFrames] = {};
            for (int64_t c = 0; c < channels; ++c) {
                const float w = weight[c];
                const float * column = &states[static_cast<size_t>(c * kHeadFrames)];
                for (int64_t t = 0; t < kHeadFrames; ++t) {
                    sums[t] += w * column[t];
                }
            }
            const float bias = branch.bias[static_cast<size_t>(o)];
            for (int64_t t = 0; t < tile; ++t) {
                head[static_cast<size_t>((t0 + t) * outputs + o)] = sums[t] + bias;
            }
        }
    }

    // The filter frames are worked a block at a time, so the memory in use does
    // not grow with the length of the utterance.
    const int64_t block = std::min<int64_t>(filter_frames, kBlockFrames);
    std::vector<float> framed(static_cast<size_t>(block * fft));
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(block * bins));
    // The overlap-add of one block: it starts at the first sample of the block's first frame.
    // Samples no later frame reaches are divided by the window sums and passed to `out` after
    // each block, so this does not grow with the utterance either.
    std::vector<float> sum(static_cast<size_t>(block * hop + fft), 0.0F);
    // The log response of the envelope of each acoustic frame a block needs, without its gains:
    // the real FFT of the liftered cepstrum, as linear in the cepstrum as the transform it replaces.
    const int64_t span = block / steps + 2;
    std::vector<float> cepstra(static_cast<size_t>(span * fft), 0.0F);
    std::vector<std::complex<float>> log_response(static_cast<size_t>(span * bins));
    std::vector<std::complex<float>> phase(static_cast<size_t>(bins));
    std::vector<float> level(static_cast<size_t>(bins));
    const auto plan = audio::get_real_fft_plan(static_cast<size_t>(fft));
    const audio::TensorStrideBytes real_strides{static_cast<std::ptrdiff_t>(fft * static_cast<int64_t>(sizeof(float))),
                                                static_cast<std::ptrdiff_t>(sizeof(float))};
    const audio::TensorStrideBytes complex_strides{static_cast<std::ptrdiff_t>(bins * static_cast<int64_t>(sizeof(std::complex<float>))),
                                                   static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};

    // The sum of squared windows at a sample depends only on where it sits: within the first
    // frame, on its place in that frame; after it, on its place in the hop, as one more frame
    // starts every hop. It is a table, added up in the order the frames come.
    const float * window = branch.window.data();
    auto squares = [&](int64_t at) {
        float value = 0.0F;
        for (int64_t i = std::max<int64_t>(0, (at - fft + hop) / hop); i <= std::min(filter_frames - 1, at / hop); ++i) {
            const float w = window[at - i * hop];
            value += w * w;
        }
        return value;
    };
    std::vector<float> squared(static_cast<size_t>(fft));
    for (int64_t at = 0; at < fft; ++at) {
        squared[static_cast<size_t>(at)] = squares(at);
    }
    // Every frame over the samples before `complete` exists; any after it are added up one by one.
    const int64_t complete = std::max<int64_t>(std::min(samples, filter_frames * hop - fft / 2), 0);
    // Output samples whose frame positions are in [from, to); the block's sums start at `base`.
    auto finish = [&](int64_t from, int64_t to, int64_t base) {
        int64_t n = std::max<int64_t>(from - fft / 2, 0);
        const int64_t end = std::min(to - fft / 2, samples);
        while (n < std::min(end, complete)) {
            const int64_t at = n + fft / 2;
            const int64_t place = at < fft ? at : fft - hop + (at - fft) % hop;
            const int64_t length = std::min(at < fft ? fft - at : fft - place, std::min(end, complete) - n);
            const float * norm = &squared[static_cast<size_t>(place)];
            const float * total = &sum[static_cast<size_t>(at - base)];
            float * into = &out[static_cast<size_t>(n)];
            for (int64_t j = 0; j < length; ++j) {
                into[j] += norm[j] > 1e-11F ? total[j] / norm[j] : total[j];
            }
            n += length;
        }
        for (; n < end; ++n) {
            const int64_t at = n + fft / 2;
            const float norm = squares(at);
            const float total = sum[static_cast<size_t>(at - base)];
            out[static_cast<size_t>(n)] += norm > 1e-11F ? total / norm : total;
        }
    };
    const int64_t last = frames - 1;
    for (int64_t first = 0; first < filter_frames; first += block) {
        const int64_t count = std::min(block, filter_frames - first);
        const audio::TensorShape frame_shape{static_cast<size_t>(count), static_cast<size_t>(fft)};

        // The source in windowed frames, centred, with half a window of zeros after it. Only the
        // part of each frame outside the source is zeroed, so the inner loop has no tests.
        for (int64_t i = 0; i < count; ++i) {
            const int64_t start = (first + i) * hop - fft / 2;
            const int64_t begin = std::min(std::max<int64_t>(-start, 0), fft);
            const int64_t end = std::max(std::min(samples - start, fft), begin);
            float * frame = &framed[static_cast<size_t>(i * fft)];
            std::fill(frame, frame + begin, 0.0F);
            const float * window = branch.window.data() + begin;
            const float * from = source.data() + (start + begin);
            float * to = frame + begin;
            for (int64_t n = 0; n < end - begin; ++n) {
                to[n] = from[n] * window[n];
            }
            std::fill(frame + end, frame + fft, 0.0F);
        }
        plan->forward(frame_shape, real_strides, complex_strides, 1, framed.data(), spectrum.data(), 1.0F, kFftThreads);

        // The envelopes of the acoustic frames this block reads, transformed together.
        const int64_t final_sub = steps * frames - 1;
        const int64_t low = std::min(first, final_sub) / steps;
        const int64_t high = std::min((std::min(first + count - 1, final_sub) + 1) / steps, last);
        const int64_t envelopes = high - low + 1;
        for (int64_t e = 0; e < envelopes; ++e) {
            const float * a = &head[static_cast<size_t>((low + e) * outputs)];
            float * frame = &cepstra[static_cast<size_t>(e * fft)];
            for (int64_t c = 0; c < coefficients; ++c) {
                frame[branch.positions[static_cast<size_t>(c)]] = a[c] * branch.lifter[static_cast<size_t>(c)];
            }
        }
        const audio::TensorShape envelope_shape{static_cast<size_t>(envelopes), static_cast<size_t>(fft)};
        plan->forward(envelope_shape, real_strides, complex_strides, 1, cepstra.data(), log_response.data(), 1.0F, kFftThreads);

        // Each filter frame: response exp(log response + gain), times the spectrum. The harmonic
        // envelope is interpolated between frames, the noise one is held; its phase is shared by
        // the filter frames with the same envelope.
        int64_t phase_key = -1;
        std::int32_t shared = 0;
        for (int64_t i = 0; i < count; ++i) {
            const int64_t sub = std::min(first + i, final_sub);
            const int64_t here = sub / steps;
            const int64_t following = interpolate ? std::min((sub + 1) / steps, last) : here;
            const std::complex<float> * a = &log_response[static_cast<size_t>((here - low) * bins)];
            const std::complex<float> * b = &log_response[static_cast<size_t>((following - low) * bins)];
            const float gain = head[static_cast<size_t>(here * outputs + coefficients + sub % steps)];
            const int64_t key = here * frames + following;
            if (key != phase_key) {
                std::int32_t outside = 0;  // a count, not a flag: || would branch
                for (int64_t k = 0; k < bins; ++k) {
                    const float angle = 0.5F * (a[k].imag() + b[k].imag());
                    float s = 0.0F;
                    float c = 0.0F;
                    sincos_bounded(angle, s, c);
                    phase[static_cast<size_t>(k)] = std::complex<float>(c, s);
                    outside += std::fabs(angle) > kSincosRange ? 1 : 0;
                }
                // Phases this large do not come out of a trained head; they get the library's sines.
                if (outside > 0) {
                    for (int64_t k = 0; k < bins; ++k) {
                        const float angle = 0.5F * (a[k].imag() + b[k].imag());
                        phase[static_cast<size_t>(k)] = std::complex<float>(std::cos(angle), std::sin(angle));
                    }
                }
                // A held envelope serves `steps` filter frames that differ only by their gain:
                // exp(a + gain) is then exp(a) exp(gain), with exp(a) worked out once.
                shared = 0;
                if (!interpolate) {
                    for (int64_t k = 0; k < bins; ++k) {
                        const float value = a[k].real();
                        level[static_cast<size_t>(k)] = exp_bounded(std::min(std::max(value, -kLevelRange), kLevelRange));
                        shared += std::fabs(value) > kLevelRange ? 1 : 0;
                    }
                    shared = shared == 0 ? 1 : 0;
                }
                phase_key = key;
            }
            std::complex<float> * row = &spectrum[static_cast<size_t>(i * bins)];
            // That holds while no bin is clamped, which takes one pass of comparisons to know.
            std::int32_t clamped = shared > 0 && std::fabs(gain) <= kLevelRange ? 0 : 1;
            if (clamped == 0) {
                for (int64_t k = 0; k < bins; ++k) {
                    const float log_magnitude = a[k].real() + gain;
                    clamped += log_magnitude < -14.0F || log_magnitude > 6.0F ? 1 : 0;
                }
            }
            if (clamped == 0) {
                const float scale = exp_bounded(gain);
                for (int64_t k = 0; k < bins; ++k) {
                    const float magnitude = level[static_cast<size_t>(k)] * scale;
                    const float re = magnitude * phase[static_cast<size_t>(k)].real();
                    const float im = magnitude * phase[static_cast<size_t>(k)].imag();
                    row[k] = std::complex<float>(row[k].real() * re - row[k].imag() * im, row[k].real() * im + row[k].imag() * re);
                }
                continue;
            }
            for (int64_t k = 0; k < bins; ++k) {
                const float log_magnitude = 0.5F * (a[k].real() + b[k].real()) + gain;
                const float magnitude = exp_bounded(std::min(std::max(log_magnitude, -14.0F), 6.0F));
                // written out: std::complex products check for infinities on every call
                const float re = magnitude * phase[static_cast<size_t>(k)].real();
                const float im = magnitude * phase[static_cast<size_t>(k)].imag();
                row[k] = std::complex<float>(row[k].real() * re - row[k].imag() * im, row[k].real() * im + row[k].imag() * re);
            }
        }

        // Back to samples: overlap-add of the windowed frames over the sum of squared windows.
        plan->inverse(frame_shape, complex_strides, real_strides, 1, spectrum.data(), framed.data(), 1.0F / static_cast<float>(fft), kFftThreads);
        for (int64_t i = 0; i < count; ++i) {
            float * into = &sum[static_cast<size_t>(i * hop)];
            const float * frame = &framed[static_cast<size_t>(i * fft)];
            for (int64_t n = 0; n < fft; ++n) {
                into[n] += frame[n] * window[n];
            }
        }
        // Samples before the next block's first frame are complete; the rest moves to the front.
        const int64_t base = first * hop;
        const int64_t next = (first + count) * hop;
        const bool final_block = first + count >= filter_frames;
        finish(base, final_block ? samples + fft / 2 : next, base);
        if (!final_block) {
            std::copy(sum.begin() + (next - base), sum.begin() + (next - base) + (fft - hop), sum.begin());
            std::fill(sum.begin() + (fft - hop), sum.end(), 0.0F);
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
