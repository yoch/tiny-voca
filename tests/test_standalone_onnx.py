import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from test_export import small_model, utterance

from tiny_voca.export import Exportable, _FrameOwners, export


class StandaloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scan_counts_duplicate_endpoints(self):
        durations = torch.tensor([[0, 3, 0, 1, 0]])
        ends = durations.cumsum(1)
        frame = torch.arange(4)[None]
        expected = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        self.assertEqual(expected.tolist(), [[1, 1, 1, 3]])
        torch.testing.assert_close(_FrameOwners.apply(ends, frame), expected)

    def test_packed_synthesis_preserves_short_and_long_crops(self):
        model = small_model()
        original = Exportable(model)
        variants = [
            Exportable(model, packed=True),
            Exportable(model, packed=True, frame_ola=True),
        ]
        rng = torch.Generator().manual_seed(3)
        for name, fft, hop in (("synth", 512, 128), ("noise", 256, 64)):
            for samples in (256, 768, 2304):
                signal = torch.randn(1, samples, generator=rng)
                shape = (1, samples // hop + 3, fft // 2 + 1)
                real = torch.randn(shape, generator=rng)
                imaginary = torch.randn(shape, generator=rng)
                target = original._filter(signal, real, imaginary, name, fft, hop)
                for candidate in variants:
                    result = candidate._packed_filter(
                        signal, real[:, :-1], imaginary[:, :-1], name, fft, hop
                    )
                    torch.testing.assert_close(result, target, rtol=1e-5, atol=3e-6)

    def test_export_only_standard_ops_and_dynamic_lengths(self):
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
            baseline = export(model, root / "base.onnx", external_noise=True)
            candidate = export(
                model,
                root / "packed.onnx",
                external_noise=True,
                omit_mask=True,
                packed=True,
                scan_owners=True,
                frame_ola=True,
            )
            graph = onnx.load(candidate)
            onnx.checker.check_model(graph)
            self.assertTrue(all(n.domain in ("", "ai.onnx") for n in graph.graph.node))
            self.assertFalse(any(n.op_type == "IsNaN" for n in graph.graph.node))
            self.assertTrue(
                any(n.op_type == "ScatterElements" for n in graph.graph.node)
            )
            reference, optimized = (
                ort.InferenceSession(
                    str(p), options, providers=["CPUExecutionProvider"]
                )
                for p in (baseline, candidate)
            )
            for length in (1, 11, 45):
                ids, stress = utterance(length)
                data = {
                    "tokens": ids.numpy(),
                    "stress": stress.numpy(),
                    "voice": np.array([1]),
                    "speed": np.array([1.0], dtype=np.float32),
                    "pitch_range": np.array([1.0], dtype=np.float32),
                    "pitch_shift": np.array([0.0], dtype=np.float32),
                    "noise": np.random.default_rng(4).standard_normal(
                        (1, 200000), dtype=np.float32
                    ),
                }
                a, da = reference.run(None, data)
                b, db = optimized.run(None, data)
                np.testing.assert_array_equal(da, db)
                self.assertEqual(a.shape, b.shape)
                self.assertLess(
                    np.linalg.norm(a - b), 1e-5 * max(np.linalg.norm(a), 1e-12)
                )


if __name__ == "__main__":
    unittest.main()
