"""E04 experiments only. Do not promote without CPU/Web benchmark evidence."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from tiny_voca.export import Exportable as DenseExportable, INPUTS, OUTPUTS, example_inputs

class _DFT(torch.autograd.Function):
    """Bridge the legacy exporter to standard ONNX DFT (no custom runtime op).

    Opset 17 requires a full spectrum for the inverse transform.
    Complex values are represented by a final dimension of size two.
    """

    @staticmethod
    def forward(ctx, value, inverse, size):
        if inverse:
            return torch.view_as_real(torch.fft.ifft(torch.view_as_complex(value.contiguous()), dim=-1))
        return torch.view_as_real(torch.fft.rfft(value[..., 0], dim=-1))

    @staticmethod
    def symbolic(g, value, inverse, size):
        attributes = {"inverse_i": int(inverse), "onesided_i": int(not inverse)}
        if g.opset >= 20:
            axis = g.op("Constant", value_t=torch.tensor(-2, dtype=torch.int64))
            return g.op("DFT", value, g.op("Constant", value_t=torch.tensor(size, dtype=torch.int64)), axis, **attributes)
        return g.op("DFT", value, axis_i=-2, **attributes)


class _RadixFFT(nn.Module):
    """Fixed-size radix-2 butterflies with real float32 ONNX operations."""
    def __init__(self, size):
        super().__init__()
        if size < 2 or size & (size - 1):
            raise ValueError("radix2 requires a power-of-two FFT size; use dense fallback")
        self.size = size
        bits = size.bit_length() - 1
        reverse = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(size)]
        self.register_buffer("reverse", torch.tensor(reverse), persistent=False)
        for stage in range(1, bits + 1):
            angle = torch.arange(2**(stage-1), dtype=torch.float64) * (-2 * math.pi / 2**stage)
            self.register_buffer(f"cos_{stage}", angle.cos().float(), persistent=False)
            self.register_buffer(f"sin_{stage}", angle.sin().float(), persistent=False)

    def forward(self, real, imaginary, inverse=False):
        real, imaginary = real[..., self.reverse], imaginary[..., self.reverse]
        for stage in range(1, self.size.bit_length()):
            size = 2**stage
            r, i = real.reshape(1,-1,size), imaginary.reshape(1,-1,size)
            ar, br = r[..., :size//2], r[..., size//2:]
            ai, bi = i[..., :size//2], i[..., size//2:]
            co, si = getattr(self, f"cos_{stage}"), getattr(self, f"sin_{stage}")
            if inverse:
                si = -si
            tr, ti = br * co - bi * si, br * si + bi * co
            real = torch.cat((ar + tr, ar - tr), dim=-1)
            imaginary = torch.cat((ai + ti, ai - ti), dim=-1)
        real, imaginary = real.reshape(1,-1,self.size), imaginary.reshape(1,-1,self.size)
        if inverse:
            return real / self.size, imaginary / self.size
        return real, imaginary


def _overlap_add(frames, hop):
    """Four-way overlap, using only reshape, slice, pad and addition on Web."""
    return sum(F.pad(frames[..., i * hop:(i + 1) * hop].reshape(1, -1),
                     (i * hop, (3 - i) * hop)) for i in range(4))


class Exportable(DenseExportable):
    """Experimental FFT paths; the production exporter remains unchanged."""
    def __init__(self, model, external_noise=False, fft_backend="dense"):
        if fft_backend not in ("dense", "dft", "radix2"):
            raise ValueError("unknown FFT backend")
        super().__init__(model, external_noise)
        self.fft_backend = fft_backend
        for name, window in (("synth", model.vocoder.synth_window), ("noise", model.vocoder.noise_window)):
            self.register_buffer(f"{name}_window", window, persistent=False)
            if fft_backend == "radix2":
                self.add_module(f"{name}_fft", _RadixFFT(len(window)))

    def _filter(self, source, real, imaginary, name, fft, hop):
        if self.fft_backend == "dense":
            return super()._filter(source, real, imaginary, name, fft, hop)
        return self._filter_fft(source, real, imaginary, name, fft, hop)

    def _filter_fft(self, source, real, imaginary, name, fft, hop):
        samples = source.shape[1]
        padded = F.pad(source, (fft // 2, fft))
        starts = torch.arange(samples // hop + 3, device=source.device) * hop
        indices = starts[:, None] + torch.arange(fft, device=source.device)
        window = getattr(self, f"{name}_window")
        frames = padded[:, indices] * window
        if self.fft_backend == "radix2":
            sr, si = getattr(self, f"{name}_fft")(frames, torch.zeros_like(frames))
            sr, si = sr[..., :fft//2+1], si[..., :fft//2+1]
        else:
            spectrum = _DFT.apply(frames[..., None], False, fft)
            sr, si = spectrum[..., 0], spectrum[..., 1]
        pr, pi = sr * real - si * imaginary, sr * imaginary + si * real
        # Reconstruct Hermitian symmetry; inverse+onesided is invalid in opset 17.
        pr = torch.cat((pr, pr[..., 1:-1].flip(-1)), dim=-1)
        pi = torch.cat((pi, -pi[..., 1:-1].flip(-1)), dim=-1)
        if self.fft_backend == "radix2":
            wave = getattr(self, f"{name}_fft")(pr, pi, True)[0] * window
        else:
            wave = _DFT.apply(torch.stack((pr, pi), dim=-1), True, fft)[..., 0] * window
        wave = _overlap_add(wave, hop)
        envelope = _overlap_add((window * window).expand_as(frames), hop)
        # Crop before division so the zero-valued window endpoints are never divided.
        return wave[:, fft // 2:fft // 2 + samples] / envelope[:, fft // 2:fft // 2 + samples]

def export(model, path, external_noise=False, opset=17, fft_backend="dense"):
    if opset < 17 and fft_backend != "dense":
        raise ValueError("FFT experiments require opset >= 17")
    wrapper = Exportable(model, external_noise, fft_backend).eval()
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
