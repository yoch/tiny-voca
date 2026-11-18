"""ONNX export of free-running synthesis.

    python -m tiny_voca.export --checkpoint inference.pt --output tiny-voca.onnx

The exported graph speaks one utterance: phoneme ids and stress in, waveform
out. Text is turned into ids outside the graph (``tiny_voca.text.phonemize``).

``Exportable`` computes the same audio as ``TinyVoca.synthesize`` for a single
utterance, written with operations ONNX has: the short-time Fourier transforms
of the source-filter head become real convolutions with fixed kernels, the
complex filter responses are kept as separate real and imaginary parts, and
the frame expansion uses a comparison instead of a sorted search.

Inputs
    tokens       int64 [1, N]   phoneme ids, with the edge tokens
    stress       int64 [1, N]
    voice        int64 [1]      index into ``voices`` in the metadata
    speed        float32 [1]    1.0 = as predicted, larger is faster
    pitch_shift  float32 [1]    added to log F0 (0.058 is about a semitone)
    pitch_range  float32 [1]    scales pitch movement around the mean
Outputs
    audio        float32 [1, samples]
    durations    int64 [1, N]   frames per token (``hop`` samples each)
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .infer import load_model, match_frontend
from .text import SYMBOLS, frontend_fingerprint, frontend_identity

INPUTS = ("tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range")
OUTPUTS = ("audio", "durations")


def _cepstrum_matrices(positions, fft):
    """Real and imaginary parts of rfft for unit coefficients at ``positions``."""
    buffer = torch.zeros(len(positions), fft, dtype=torch.float64)
    buffer[torch.arange(len(positions)), torch.tensor(positions)] = 1
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return spectrum.real.float(), spectrum.imag.float()


def _analysis_kernels(window):
    """Convolution kernels [bins, 1, fft] giving the real and imaginary STFT."""
    fft = len(window)
    spectrum = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
    return spectrum.real.T[:, None].float().contiguous(), spectrum.imag.T[:, None].float().contiguous()


def _synthesis_kernel(window):
    """Transposed-convolution kernel [2 * bins, 1, fft]: windowed inverse rfft
    of the real parts (first half of the channels) and imaginary parts."""
    fft = len(window)
    bins = fft // 2 + 1
    unit = torch.eye(bins, dtype=torch.complex128)
    real = torch.fft.irfft(unit, n=fft, dim=-1)
    imaginary = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
    return (torch.cat((real, imaginary)) * window.double())[:, None].float().contiguous()


class Exportable(nn.Module):
    """Single-utterance synthesis in ONNX-friendly operations."""

    def __init__(self, model, external_noise=False, omit_mask=True):
        super().__init__()
        self.model = model
        self.external_noise = external_noise
        self.omit_mask = omit_mask
        c = model.config
        v = model.vocoder
        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        for name, (real, imaginary) in (("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
                                        ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft))):
            self.register_buffer(f"{name}_real", real, persistent=False)
            self.register_buffer(f"{name}_imaginary", imaginary, persistent=False)
        for name, window in (("synth", v.synth_window), ("noise", v.noise_window)):
            real, imaginary = _analysis_kernels(window)
            self.register_buffer(f"{name}_analysis_real", real, persistent=False)
            self.register_buffer(f"{name}_analysis_imaginary", imaginary, persistent=False)
            self.register_buffer(f"{name}_synthesis", _synthesis_kernel(window), persistent=False)
            self.register_buffer(f"{name}_envelope", (window.double() ** 2).float()[None, None].contiguous(), persistent=False)
        self.register_buffer("ramp", torch.arange(c.hop, dtype=torch.float32) / c.hop, persistent=False)
        self.phase_steps = 1 << 24

    def _filter(self, source, real, imaginary, name, fft, hop):
        """Multiply the STFT of ``source`` [1, samples] by a response given as
        real and imaginary parts [1, frames, bins], and resynthesize."""
        samples = source.shape[1]
        padded = F.pad(source[:, None], (fft // 2, fft))
        s_real = F.conv1d(padded, getattr(self, f"{name}_analysis_real"), stride=hop)
        s_imaginary = F.conv1d(padded, getattr(self, f"{name}_analysis_imaginary"), stride=hop)
        real, imaginary = real.transpose(1, 2), imaginary.transpose(1, 2)
        product = torch.cat((s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real), dim=1)
        wave = F.conv_transpose1d(product, getattr(self, f"{name}_synthesis"), stride=hop)
        envelope = F.conv_transpose1d(torch.ones_like(s_real[:, :1]), getattr(self, f"{name}_envelope"), stride=hop)
        return (wave / envelope)[:, 0, fft // 2:fft // 2 + samples]

    @staticmethod
    def _response(cepstrum, real, imaginary):
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _pulses(self, f0):
        """``pulse_train`` for one utterance, without 64-bit floats (the web
        runtime has none). The phase is summed inside each frame in float32,
        where it stays below a few cycles, and across frames as 24-bit integer
        fractions of a cycle, which wrap exactly however long the utterance is."""
        c = self.model.config
        knots = torch.log(f0.clamp_min(1.0))
        knots = torch.cat((knots, knots[:, -1:]), dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        frequency = torch.exp(low * (1 - self.ramp) + high * self.ramp)
        within = torch.cumsum(frequency / c.sample_rate, dim=2)
        whole_frame = within[:, :, -1]
        steps = torch.round((whole_frame - torch.floor(whole_frame)) * self.phase_steps).long()
        start = torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype) / self.phase_steps
        cycles = within + start[:, :, None]
        theta = (2 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        frequency = frequency.reshape(1, -1)
        harmonics = (c.harmonic_limit * c.sample_rate / frequency).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denominator = 2 * torch.sin(theta / 2)
        tiny = denominator.abs() < 1e-6
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
        wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
        return wave * torch.sqrt(2.0 / harmonics)

    def _vocoder(self, hidden, f0, total, noise):
        c = self.model.config
        v = self.model.vocoder
        last = total - 1
        # Harmonic branch.
        h = v.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        steps = v.harmonic_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        here = torch.div(sub, steps, rounding_mode="floor")
        following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
        envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
        envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
        gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
        envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
        harmonic_real, harmonic_imaginary = self._response(envelope, self.harmonic_real, self.harmonic_imaginary)
        # Noise branch.
        n = v.noise(hidden)
        steps = v.noise_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        shape = n[..., :c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor")) * v.noise_lifter
        gain = n[..., c.noise_cep:].flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)
        pulses = self._pulses(f0)
        if noise is None:
            noise = torch.randn_like(pulses)
        else:
            noise = noise.index_select(1, torch.arange(c.hop * total, device=hidden.device))
        return (self._filter(pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop)
                + self._filter(noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop))

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range, noise=None):
        m = self.model
        c = m.config
        token_mask = None if self.omit_mask else torch.ones_like(tokens, dtype=torch.bool)
        hidden = m.encoder(m.tokens(tokens) + m.stress(stress) + m.voices(voice)[:, None], token_mask)
        # Durations.
        output = m.duration(hidden, token_mask)
        frames = torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (output[..., 1] > 0)
        durations = torch.where(present, frames, torch.zeros_like(frames))
        # Repeat token states by duration.
        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(total, device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(features) + m.frame_voices(voice)[:, None]
        frame_mask = None if self.omit_mask else torch.ones_like(frame, dtype=torch.bool)
        # Pitch, voicing, energy.
        prosody_hidden = m.prosody_stack(expanded, frame_mask)
        prosody = m.prosody_head(prosody_hidden)
        vowel = m.vowel_tokens[tokens].gather(1, index)
        voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
        voiced = (voicing > 0).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1)
        acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask)
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
        return self._vocoder(acoustic, f0, total, noise), durations


def example_inputs(length=24):
    tokens = torch.randint(5, len(SYMBOLS), (1, length))
    one = lambda value: torch.tensor([value], dtype=torch.float32)
    return tokens, torch.zeros_like(tokens), torch.tensor([0]), one(1.0), one(0.0), one(1.0)


def export(model, path, external_noise=False, opset=17, omit_mask=True):
    wrapper = Exportable(model, external_noise, omit_mask=omit_mask).eval()
    inputs = example_inputs()
    names = list(INPUTS)
    axes = {"tokens": {1: "tokens"}, "stress": {1: "tokens"}, "audio": {1: "samples"}, "durations": {1: "tokens"}}
    if external_noise:
        inputs = (*inputs, torch.randn(1, model.config.hop * model.config.max_frames))
        names.append("noise")
        axes["noise"] = {1: "noise_samples"}
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=list(OUTPUTS), dynamic_axes=axes,
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def metadata(model, checkpoint):
    c = model.config
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {"sample_rate": c.sample_rate, "hop": c.hop, "voices": list(c.speakers), "symbols": list(SYMBOLS),
            "max_tokens": c.max_tokens, "max_frames": c.max_frames, "step": state.get("step"),
            "frontend": frontend_identity().get("phonemizer", "espeak"), "frontend_fingerprint": frontend_fingerprint(),
            "inputs": list(INPUTS), "outputs": list(OUTPUTS)}


def main():
    parser = argparse.ArgumentParser(description="Export a tiny-voca inference checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-noise", action="store_true", help="Take the noise signal as an input (for exact comparisons)")
    parser.add_argument("--keep-mask", action="store_true", help="Keep redundant all-true attention masks in single-utterance ONNX")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.output, args.external_noise, args.opset, omit_mask=not args.keep_mask)
    args.output.with_suffix(".json").write_text(json.dumps(metadata(model, args.checkpoint), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) and {args.output.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
