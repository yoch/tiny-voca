"""E04 experimental ONNX boundaries (unmasked, fused FFT, compact FFT) and C DSP wrapper."""

import ctypes as ct
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import torch

from tiny_voca.export import Exportable, INPUTS, example_inputs

FLOAT_PTR = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
DSP_SOURCE = Path(__file__).with_name("dsp.c")

EXPECTED_VOCODER_CONFIG = (24000, 256, 512, 128, 256, 64, 80, 16, 32, 0.47)


def verify_vocoder_config(model) -> None:
    """Assert the model uses the fixed FFT sizes/cepstra supported by dsp.c."""
    c = model.config
    actual = (
        c.sample_rate, c.hop, c.synth_fft, c.synth_hop,
        c.noise_fft, c.noise_hop, c.cep_causal, c.cep_anti,
        c.noise_cep, c.harmonic_limit,
    )
    if actual != EXPECTED_VOCODER_CONFIG:
        raise ValueError(f"dsp.c requires {EXPECTED_VOCODER_CONFIG}, got {actual}")


def compile_dsp(output_so: Path, cc: str = "gcc", march: str = "native") -> Path:
    """Compile scripts/e04/dsp.c into a shared library without illegal ISA flags."""
    output_so.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cc, "-O3", f"-march={march}", "-ffp-contract=off",
        "-fPIC", "-shared", str(DSP_SOURCE), "-lm", "-o", str(output_so),
    ]
    subprocess.run(cmd, check=True)
    return output_so


def extract_vocoder_constants(model) -> list[list[float]]:
    """Extract window and cepstral lifter arrays required by dsp.c initialize()."""
    verify_vocoder_config(model)
    v = model.vocoder
    fields = ("synth_window", "noise_window", "causal_lifter", "anti_lifter", "noise_lifter")
    return [getattr(v, name).detach().cpu().numpy().astype(np.float32).tolist() for name in fields]


class FusedExportable(Exportable):
    """Export neural stacks, cepstral response, and pulses; emit STFT filter inputs."""

    def _filter(self, source, real, imaginary, name, fft, hop):
        self.captured.extend((source, real, imaginary))
        return torch.zeros_like(source)

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range, noise=None):
        self.captured = []
        _, durations = super().forward(tokens, stress, voice, speed, pitch_shift, pitch_range, noise=noise)
        return (*self.captured, durations)


class CompactExportable(Exportable):
    """Export transformer stacks, linear vocoder heads, and float32 pulses only."""

    def __init__(self, model, omit_mask=True):
        super().__init__(model, external_noise=True, omit_mask=omit_mask)

    def _vocoder(self, hidden, f0, total, noise):
        v = self.model.vocoder
        return v.harmonic(hidden), v.noise(hidden), self._pulses(f0)

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range, noise=None):
        (harmonic, noise_head, pulses), durations = super().forward(
            tokens, stress, voice, speed, pitch_shift, pitch_range, noise=None,
        )
        return harmonic, noise_head, pulses, durations


def export_fused(model, path: Path, omit_mask: bool = True, opset: int = 17) -> Path:
    """Export FusedExportable (no dense STFT Conv/ConvTranspose nodes)."""
    verify_vocoder_config(model)
    wrapper = FusedExportable(model, external_noise=True, omit_mask=omit_mask).eval()
    inputs = (*example_inputs(), torch.randn(1, model.config.hop * model.config.max_frames))
    names = [*INPUTS, "noise"]
    outputs = ("pulses", "hr", "hi", "noise_source", "nr", "ni", "durations")
    axes = {"tokens": {1: "tokens"}, "stress": {1: "tokens"}, "noise": {1: "noise_samples"}, "durations": {1: "tokens"}}
    for out_name in outputs[:-1]:
        axes[out_name] = {1: "samples" if out_name in ("pulses", "noise_source") else f"frames_{out_name[0]}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=list(outputs),
                          dynamic_axes=axes, opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)
    return path


def export_compact(model, path: Path, omit_mask: bool = True, opset: int = 17) -> Path:
    """Export CompactExportable (no STFT convs and no cepstral DFT/trig matrices)."""
    verify_vocoder_config(model)
    wrapper = CompactExportable(model, omit_mask=omit_mask).eval()
    inputs = example_inputs()
    outputs = ("harmonic", "noise_head", "pulses", "durations")
    axes = {
        "tokens": {1: "tokens"},
        "stress": {1: "tokens"},
        "harmonic": {1: "frames"},
        "noise_head": {1: "frames"},
        "pulses": {1: "samples"},
        "durations": {1: "tokens"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=list(INPUTS), output_names=list(outputs),
                          dynamic_axes=axes, opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)
    return path


class CDspVocoder:
    """Thin ctypes wrapper around compiled dsp.so reusing preallocated output buffers."""

    def __init__(self, so_path: Path, constants_json: Path, max_samples: int):
        self.lib = ct.CDLL(str(so_path))
        signatures = (
            ("initialize", 0, 5),
            ("filter", 2, 4),
            ("compact_filter", 2, 3),
            ("mix", 1, 2),
        )
        for name, n_ints, n_ptrs in signatures:
            fn = getattr(self.lib, name)
            fn.argtypes = [ct.c_int] * n_ints + [FLOAT_PTR] * n_ptrs
            fn.restype = None

        raw = json.loads(constants_json.read_text(encoding="utf-8"))
        self._constants = [np.ascontiguousarray(arr, dtype=np.float32) for arr in raw]
        self.lib.initialize(*self._constants)
        self.max_samples = max_samples
        self._harmonic = np.empty(max_samples, dtype=np.float32)
        self._noise = np.empty(max_samples, dtype=np.float32)

    def _ensure_capacity(self, samples: int) -> None:
        if samples > self.max_samples:
            self.max_samples = samples
            self._harmonic = np.empty(samples, dtype=np.float32)
            self._noise = np.empty(samples, dtype=np.float32)

    def filter_single(self, branch: int, source: np.ndarray, real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        """Run one STFT+iSTFT filter branch (`0`=synth 512/128, `1`=noise 256/64)."""
        src = np.ascontiguousarray(source.reshape(-1), dtype=np.float32)
        re = np.ascontiguousarray(real, dtype=np.float32)
        im = np.ascontiguousarray(imag, dtype=np.float32)
        samples = src.size
        self._ensure_capacity(samples)
        self.lib.filter(branch, samples, src, re, im, self._harmonic)
        return self._harmonic[:samples].copy()

    def run_fused(self, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize full waveform from FusedExportable outputs."""
        pulses, hr, hi, noise_src, nr, ni, durations = parts
        p = np.ascontiguousarray(pulses.reshape(-1), dtype=np.float32)
        n = np.ascontiguousarray(noise_src.reshape(-1), dtype=np.float32)
        samples = p.size
        self._ensure_capacity(samples)
        self.lib.filter(0, samples, p, np.ascontiguousarray(hr, np.float32), np.ascontiguousarray(hi, np.float32), self._harmonic)
        self.lib.filter(1, samples, n, np.ascontiguousarray(nr, np.float32), np.ascontiguousarray(ni, np.float32), self._noise)
        self.lib.mix(samples, self._harmonic, self._noise)
        return self._harmonic[:samples][None].copy(), durations

    def run_compact(self, parts: list[np.ndarray], noise_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize full waveform from CompactExportable outputs + raw noise stream."""
        harmonic, noise_head, pulses, durations = parts
        frames = harmonic.shape[1]
        samples = frames * 256
        self._ensure_capacity(samples)
        p = np.ascontiguousarray(pulses.reshape(-1), dtype=np.float32)
        n = np.ascontiguousarray(noise_full[:samples], dtype=np.float32)
        h_head = np.ascontiguousarray(harmonic, dtype=np.float32)
        n_head = np.ascontiguousarray(noise_head, dtype=np.float32)
        self.lib.compact_filter(0, frames, p, h_head, self._harmonic)
        self.lib.compact_filter(1, frames, n, n_head, self._noise)
        self.lib.mix(samples, self._harmonic, self._noise)
        return self._harmonic[:samples][None].copy(), durations
