"""Numerical acceptance tests for E04 experiments, not a production rollout."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fft_candidates import Exportable, export
from test_export import small_model, utterance


class FFTFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = small_model()

    def test_deterministic_filters_match_dense_and_native(self):
        from tiny_voca.model import filtered
        dense = Exportable(self.model, fft_backend="dense")
        fft_wrapper = Exportable(self.model, fft_backend="radix2")
        rng = torch.Generator().manual_seed(42)
        for name, size, hop in (("synth", 512, 128), ("noise", 256, 64)):
            window = getattr(self.model.vocoder, name + "_window")
            for samples in (256, 1792, 48000, 48001):
                source = torch.randn(1, samples, generator=rng)
                frames = samples // hop + 3
                real = torch.randn(1, frames, size // 2 + 1, generator=rng) * .2
                imaginary = torch.randn(real.shape, generator=rng) * .2
                for kind in ("random", "identity", "zero", "impulse"):
                    x = source.clone()
                    r, i = real, imaginary
                    if kind == "identity":
                        r, i = torch.ones_like(real), torch.zeros_like(real)
                    if kind == "zero":
                        x.zero_()
                    if kind == "impulse":
                        x.zero_(); x[0, 0] = 1; x[0, -1] = -1
                    with torch.no_grad():
                        actual = fft_wrapper._filter(x, r, i, name, size, hop)
                        previous = dense._filter(x, r, i, name, size, hop)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, previous, atol=3e-6, rtol=2e-5)
                    if kind == "identity":
                        native = filtered(x, torch.complex(r[:, :-2], i[:, :-2]), size, hop, window)
                        torch.testing.assert_close(actual, native, atol=3e-6, rtol=2e-5)
                        torch.testing.assert_close(actual, x, atol=1e-6, rtol=1e-5)

    def test_full_onnx_parity_and_fallback(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("requires onnx and onnxruntime")
        model = self.model
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            sessions = {}
            for backend, opset in (("dense",17),("dft",17),("dft",20),("radix2",17),("radix2",20)):
                path = export(model, Path(directory)/f"{backend}-{opset}.onnx", True, opset, backend)
                graph = onnx.load(path)
                onnx.checker.check_model(graph, full_check=True)
                types = [node.op_type for node in graph.graph.node]
                self.assertEqual(types.count("DFT"), 4 if backend == "dft" else 0)
                if backend == "radix2":
                    self.assertNotIn("ConvTranspose",types)
                sessions[backend,opset] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for length, voice, speed in ((1,0,1.),(5,1,1.25),(40,0,.8)):
                tokens, stress = utterance(length)
                one = lambda v: np.array([v],dtype=np.float32)
                noise = np.random.default_rng(42).standard_normal((1, model.config.hop * model.config.max_frames)).astype(np.float32)
                feeds = dict(tokens=tokens.numpy(), stress=stress.numpy(), voice=np.array([voice],dtype=np.int64),
                             speed=one(speed),pitch_shift=one(.1),pitch_range=one(1.3),noise=noise)
                reference, durations = sessions["dense",17].run(None,feeds)
                for backend in (("dft",17),("dft",20),("radix2",17),("radix2",20)):
                    audio, actual_durations = sessions[backend].run(None,feeds)
                    np.testing.assert_array_equal(actual_durations,durations)
                    self.assertEqual(audio.shape,reference.shape)
                    error = audio.astype(np.float64)-reference
                    self.assertLess(np.abs(error).max()/np.abs(reference).max(), 2e-5)
                    self.assertGreater(10*np.log10(np.sum(reference.astype(np.float64)**2)/np.sum(error**2)), 100)

    def test_invalid_backend_and_opset(self):
        with self.assertRaises(ValueError):
            Exportable(self.model, fft_backend="unknown")
        with self.assertRaises(ValueError):
            export(self.model, "unused.onnx", opset=16, fft_backend="radix2")

