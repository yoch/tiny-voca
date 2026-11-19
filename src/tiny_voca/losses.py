import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale == 0.0:
            return torch.zeros_like(grad_output), None
        return grad_output * ctx.scale, None


def scale_gradient(value, scale):
    """Differentiable gradient gate: bit-for-bit identity in forward, scales backward gradient by ``scale``."""
    if scale == 1.0:
        return value
    return _ScaleGradient.apply(value, float(scale))


def pitch_shape_loss(
    predicted,
    target,
    reliable_mask,
    energy=None,
    energy_floor=-0.5,
    contrast_window=19,
):
    """Robust local F0 contour shape loss on reliably voiced interior frames.

    Supervises:
    - 1st difference (1-frame melodic slope, ~10.7 ms) with jump guard,
    - 3-frame syllable-scale slope (~32 ms) inside continuous reliable segments,
    - 2nd difference (local curvature / turning points),
    - local accent contrast relative to a ~200 ms reliable voiced neighborhood.

    Voiced/unvoiced edges, untrusted/interpolated frames (e.g. M1 pyin dropouts),
    low-energy tails and pitch-tracker jumps are masked out. All terms use Huber
    (Smooth L1) against the target contour so artificial oscillations are penalized.
    """
    valid = reliable_mask > 0.5
    if energy is not None:
        valid = valid & (energy > energy_floor)
    t_len = predicted.shape[1]
    zero = predicted.new_zeros((), dtype=torch.float32)
    if t_len < 3:
        return zero

    # Erode 1 frame from every voiced island boundary to exclude onset/offset transients.
    interior = torch.zeros_like(valid)
    interior[:, 1:-1] = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]

    # 1st derivative: local melodic slope (1 frame).
    d1_mask = interior[:, :-1] & interior[:, 1:] & ((target[:, 1:] - target[:, :-1]).abs() <= 0.18)
    d1_loss = masked_mean(
        F.smooth_l1_loss(predicted[:, 1:] - predicted[:, :-1], target[:, 1:] - target[:, :-1], beta=0.04, reduction="none"),
        d1_mask.float(),
    )

    # Multi-frame syllable-scale slope (3 frames ~ 32 ms) strictly inside continuous voiced islands.
    if t_len >= 5:
        d3_mask = (
            interior[:, :-3]
            & valid[:, 1:-2]
            & valid[:, 2:-1]
            & interior[:, 3:]
            & ((target[:, 3:] - target[:, :-3]).abs() <= 0.40)
        )
        d3_loss = masked_mean(
            F.smooth_l1_loss(predicted[:, 3:] - predicted[:, :-3], target[:, 3:] - target[:, :-3], beta=0.06, reduction="none"),
            d3_mask.float(),
        )
    else:
        d3_loss = zero

    # 2nd derivative: inflection / turning points on interior triplets.
    d2_mask = (
        interior[:, :-2]
        & interior[:, 1:-1]
        & interior[:, 2:]
        & ((target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]).abs() <= 0.20)
    )
    d2_loss = masked_mean(
        F.smooth_l1_loss(
            predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2],
            target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2],
            beta=0.03,
            reduction="none",
        ),
        d2_mask.float(),
    )

    # Local accent prominence relative to reliable interior voiced frames in a phonetic window (~200 ms).
    vf = interior.float()
    k = int(contrast_window)
    if k % 2 == 0:
        k += 1
    kernel = predicted.new_ones((1, 1, k))
    w_count = F.conv1d(vf[:, None], kernel, padding=k // 2)[:, 0]
    p_mean = F.conv1d((predicted * vf)[:, None], kernel, padding=k // 2)[:, 0] / w_count.clamp_min(1.0)
    t_mean = F.conv1d((target * vf)[:, None], kernel, padding=k // 2)[:, 0] / w_count.clamp_min(1.0)
    contrast_mask = interior & (w_count >= 5.0)
    contrast_loss = masked_mean(
        F.smooth_l1_loss(predicted - p_mean, target - t_mean, beta=0.06, reduction="none"),
        contrast_mask.float(),
    )

    return d1_loss + d3_loss + 0.5 * d2_loss + contrast_loss


def magnitude(wave, fft, hop, window):
    spectrum = torch.stft(wave.float(), fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    # sqrt(re^2 + im^2 + eps): finite gradient at exact zeros (padding, silence).
    return torch.view_as_real(spectrum).square().sum(-1).add(1e-9).sqrt()


def mel_filterbank(sample_rate, fft, n_mels):
    """Slaney-scale triangular filters with area normalisation, [n_mels, bins]."""
    step, knee, slope = 200.0 / 3, 1000.0, math.log(6.4) / 27

    def to_hz(mel):
        return torch.where(mel < knee / step, mel * step, knee * torch.exp((mel - knee / step) * slope))
    top = knee / step + math.log(sample_rate / 2 / knee) / slope if sample_rate / 2 > knee else sample_rate / 2 / step
    points = to_hz(torch.linspace(0, top, n_mels + 2, dtype=torch.float64))
    bins = torch.linspace(0, sample_rate / 2, fft // 2 + 1, dtype=torch.float64)
    rising = (bins[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - bins[None]) / (points[2:] - points[1:-1])[:, None]
    weights = torch.minimum(rising, falling).clamp_min(0) * (2.0 / (points[2:] - points[:-2]))[:, None]
    return weights.float()


class SpectralLoss(nn.Module):
    """Multi-resolution STFT and log-mel distances over whole utterances.

    Frames whose centre lies beyond an utterance's length are masked, so padded
    batches give the same value as the utterances would alone. No parameters.
    """

    def __init__(self, sample_rate, resolutions=((512, 128), (1024, 256), (2048, 512)), mel=(1024, 256, 80)):
        super().__init__()
        self.resolutions, self.mel = tuple(resolutions), tuple(mel)
        for fft in {fft for fft, _ in self.resolutions} | {self.mel[0]}:
            self.register_buffer(f"window_{fft}", torch.hann_window(fft), persistent=False)
        self.register_buffer("filters", mel_filterbank(sample_rate, self.mel[0], self.mel[2]), persistent=False)

    def log_mel(self, wave):
        fft, hop, _ = self.mel
        return torch.log((self.filters @ magnitude(wave, fft, hop, getattr(self, f"window_{fft}"))).clamp_min(1e-5))

    def forward(self, predicted, target, lengths):
        total = predicted.new_zeros((), dtype=torch.float32)
        for fft, hop in self.resolutions:
            window = getattr(self, f"window_{fft}")
            p, t = magnitude(predicted, fft, hop, window), magnitude(target, fft, hop, window)
            mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
            logarithmic = ((p.clamp_min(1e-5).log() - t.clamp_min(1e-5).log()).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
            # Spectral convergence per utterance; the floor keeps silent input finite.
            convergence = (((p - t).square() * mask).sum((1, 2)) + 1e-9).sqrt() / (t.square() * mask).sum((1, 2)).sqrt().clamp_min(1.0)
            total = total + logarithmic + convergence.mean()
        hop = self.mel[1]
        p, t = self.log_mel(predicted), self.log_mel(target)
        mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[:, None]
        mel = ((p - t).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
        return total / len(self.resolutions), mel


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=(1, 32, 96, 192, 384)):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0))) for a, b in zip(channels[:-1], channels[1:]))
        self.output = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wave):
        wave = F.pad(wave, (0, (-wave.shape[-1]) % self.period))
        x = wave.reshape(wave.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, fft, hop, channels=32):
        super().__init__()
        self.fft, self.hop = fft, hop
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.output = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, wave):
        x = magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]   # [B,1,frames,bins]
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class Discriminators(nn.Module):
    """Training-only waveform critics. Never included in an inference export."""

    def __init__(self, periods=(2, 3, 5, 7, 11), resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.networks = nn.ModuleList([PeriodDiscriminator(p) for p in periods]
                                      + [SpectrogramDiscriminator(fft, hop) for fft, hop in resolutions])

    def forward(self, wave):
        return [network(wave) for network in self.networks]


def discriminator_loss(real, fake):
    return sum((r-1).square().mean() + f.square().mean()
               for (r, _), (f, _) in zip(real, fake)) / len(real)


def generator_gan_loss(real, fake):
    adversarial = sum((f-1).square().mean() for f, _ in fake) / len(fake)
    matching = sum(F.l1_loss(f, r.detach())
                   for (_, rs), (_, fs) in zip(real, fake) for r, f in zip(rs, fs))
    matching = matching / sum(len(fs) for _, fs in fake)
    return adversarial, matching
