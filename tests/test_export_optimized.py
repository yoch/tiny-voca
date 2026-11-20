import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from test_export import small_model, utterance

from tiny_voca.export import Exportable, _FrameOwners, export


class OptimizedExportTest(unittest.TestCase):
    def test_frame_owners_with_zero_durations(self):
        durations = torch.tensor([[0, 3, 0, 1, 0]])
        ends = durations.cumsum(1)
        frame = torch.arange(4)[None]
        expected = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        self.assertEqual(expected.tolist(), [[1, 1, 1, 3]])
        torch.testing.assert_close(_FrameOwners.apply(ends, frame), expected)

    def test_packed_filter_matches_filter(self):
        model = small_model()
        reference, optimized = Exportable(model), Exportable(model, optimized=True)
        rng = torch.Generator().manual_seed(3)
        for name, fft, hop in (("synth", 512, 128), ("noise", 256, 64)):
            for samples in (256, 768, 2304):
                signal = torch.randn(1, samples, generator=rng)
                shape = (1, samples // hop + 3, fft // 2 + 1)
                real, imaginary = torch.randn(shape, generator=rng), torch.randn(shape, generator=rng)
                target = reference._filter(signal, real, imaginary, name, fft, hop)
                # The optimized path needs one response frame less.
                result = optimized._packed_filter(signal, real[:, :-1], imaginary[:, :-1], name, fft, hop)
                torch.testing.assert_close(result, target, rtol=1e-5, atol=3e-6)

    def test_onnx_same_audio_standard_ops(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx and onnxruntime are required")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        model = small_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = export(model, root / "base.onnx", external_noise=True, omit_mask=False)  # fr omits masks by default
            candidate = export(model, root / "optimized.onnx", external_noise=True, optimized=True)
            graph = onnx.load(candidate)
            onnx.checker.check_model(graph)
            self.assertTrue(all(n.domain in ("", "ai.onnx") for n in graph.graph.node))
            self.assertFalse(any(n.op_type == "IsNaN" for n in graph.graph.node))
            self.assertLess(len(graph.graph.node), len(onnx.load(baseline).graph.node))
            sessions = [ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"]) for p in (baseline, candidate)]
            for length in (1, 11, 45):
                ids, stress = utterance(length)
                feed = {"tokens": ids.numpy(), "stress": stress.numpy(), "voice": np.array([1]),
                        "speed": np.array([1.0], dtype=np.float32), "pitch_shift": np.array([0.0], dtype=np.float32),
                        "pitch_range": np.array([1.0], dtype=np.float32),
                        "noise": np.random.default_rng(4).standard_normal((1, 200000), dtype=np.float32)}
                (a, da), (b, db) = (session.run(None, feed) for session in sessions)
                np.testing.assert_array_equal(da, db)
                self.assertEqual(a.shape, b.shape)
                self.assertLess(np.linalg.norm(a - b), 1e-5 * max(np.linalg.norm(a), 1e-12))

    def test_optimized_needs_opset_16(self):
        with self.assertRaises(ValueError):
            export(small_model(), "unused.onnx", optimized=True, opset=15)


if __name__ == "__main__":
    unittest.main()
