"""Numerical acceptance tests for E04 experiments, not a production rollout."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fft_candidates import Exportable, export
from benchmark_fft_native import native_filter
from test_export import small_model, utterance


class FFTFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = small_model()

    def test_deterministic_filters_match_dense_and_native(self):
        from tiny_voca.model import filtered
        dense = Exportable(self.model, fft_backend="dense")
        fft_wrappers = [Exportable(self.model, fft_backend=b) for b in ("dft", "radix2")]
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
                    for fft_wrapper in fft_wrappers:
                        with torch.no_grad():
                            actual = fft_wrapper._filter(x, r, i, name, size, hop)
                            previous = dense._filter(x, r, i, name, size, hop)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual, previous, atol=3e-6, rtol=2e-5)
                        if kind == "identity":
                            native = filtered(x, torch.complex(r[:, :-2], i[:, :-2]), size, hop, window)
                            torch.testing.assert_close(actual, native, atol=3e-6, rtol=2e-5)
                            torch.testing.assert_close(actual, x, atol=1e-6, rtol=1e-5)

    def test_filter_onnx_boundaries(self):
        try:
            import onnxruntime as ort
            import onnx
        except ImportError:
            self.skipTest("requires onnx and onnxruntime")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        class Filter(torch.nn.Module):
            def __init__(self, backend, name, size, hop):
                super().__init__()
                self.wrapper = Exportable(FFTFilterTest.model, fft_backend=backend)
                self.name, self.size, self.hop = name, size, hop
            def forward(self, source, real, imaginary):
                return self.wrapper._filter(source, real, imaginary, self.name, self.size, self.hop)
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            for name, size, hop in (("synth",512,128),("noise",256,64)):
                sessions = {}
                for backend in ('dense','dft','radix2'):
                    path = Path(directory)/f'{name}-{backend}.onnx'
                    source = torch.zeros(1, size)
                    response = torch.ones(1, size//hop+3, size//2+1)
                    torch.onnx.export(Filter(backend,name,size,hop), (source,response,torch.zeros_like(response)), str(path),
                                      input_names=['source','real','imaginary'], output_names=['audio'],
                                      dynamic_axes={n:{1:'samples' if n in ('source','audio') else 'frames'} for n in ('source','real','imaginary','audio')},
                                      opset_version=17, dynamo=False)
                    sessions[backend] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
                for samples in (hop-1,hop,hop+1,size-1,size,size+1,48001):
                    shape = (1,samples//hop+3,size//2+1)
                    source = rng.standard_normal((1,samples)).astype(np.float32)
                    real = rng.standard_normal(shape).astype(np.float32)*.2
                    imaginary = rng.standard_normal(shape).astype(np.float32)*.2
                    for kind in ('random','identity','silence','zero-response','impulse'):
                        x, r, i = source.copy(), real.copy(), imaginary.copy()
                        if kind == 'identity': r.fill(1); i.fill(0)
                        if kind == 'silence': x.fill(0)
                        if kind == 'zero-response': r.fill(0); i.fill(0)
                        if kind == 'impulse':
                            x.fill(0)
                            for position in (0,hop-1,hop,size-1,samples-1):
                                if position < samples: x[0,position] = 1 if position%2 else -1
                        feeds = dict(source=x,real=r,imaginary=i)
                        reference = sessions['dense'].run(None,feeds)[0]
                        window = getattr(self.model.vocoder,name+'_window').numpy()
                        native = native_filter(x,r,i,window,hop)
                        np.testing.assert_allclose(native,reference,atol=3e-6,rtol=2e-5)
                        for backend in ('dft','radix2'):
                            actual = sessions[backend].run(None,feeds)[0]
                            self.assertEqual(actual.shape,(1,samples))
                            self.assertTrue(np.isfinite(actual).all())
                            np.testing.assert_allclose(actual,reference,atol=3e-6,rtol=2e-5)

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
        from fft_candidates import _RadixFFT
        with self.assertRaises(ValueError):
            _RadixFFT(300)
        with self.assertRaises(ValueError):
            Exportable(self.model, fft_backend="unknown")
        with self.assertRaises(ValueError):
            export(self.model, "unused.onnx", opset=16, fft_backend="radix2")
