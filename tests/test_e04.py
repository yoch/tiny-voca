"""Numerical parity & graph-audit tests for E04 (unmasked SDPA & C packed-real FFT DSP)."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from e04.runtime import (
    CDspVocoder,
    compile_dsp,
    export_compact,
    export_fused,
    extract_vocoder_constants,
)
from test_export import small_model, utterance
from tiny_voca.export import Exportable, INPUTS, export


def _snr_db(ref: np.ndarray, cand: np.ndarray) -> float:
    num = float(np.sum(ref.astype(np.float64) ** 2))
    den = float(np.sum((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if den == 0.0:
        return float("inf")
    return float(10.0 * np.log10(num / den))


class E04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = small_model()

    def test_unmasked_export_removes_isnan_and_matches_masked(self):
        import onnx
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p_masked = export(self.model, d / "masked.onnx", external_noise=True, omit_mask=False)
            p_unmasked = export(self.model, d / "unmasked.onnx", external_noise=True, omit_mask=True)

            g_m = onnx.load(str(p_masked))
            g_u = onnx.load(str(p_unmasked))
            cm = Counter(n.op_type for n in g_m.graph.node)
            cu = Counter(n.op_type for n in g_u.graph.node)
            self.assertGreater(cm["IsNaN"], 0)
            self.assertEqual(cu["IsNaN"], 0)
            self.assertLess(cu["Where"], cm["Where"])

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sm = ort.InferenceSession(str(p_masked), opts, providers=["CPUExecutionProvider"])
            su = ort.InferenceSession(str(p_unmasked), opts, providers=["CPUExecutionProvider"])

            tokens, stress = utterance()
            rng = np.random.default_rng(0)
            noise = rng.standard_normal((1, self.model.config.hop * self.model.config.max_frames), dtype=np.float32)
            feed = {
                "tokens": tokens.numpy(),
                "stress": stress.numpy(),
                "voice": np.array([0], dtype=np.int64),
                "speed": np.array([1.0], dtype=np.float32),
                "pitch_shift": np.array([0.0], dtype=np.float32),
                "pitch_range": np.array([1.0], dtype=np.float32),
                "noise": noise,
            }
            am, dm = sm.run(None, feed)
            au, du = su.run(None, feed)
            np.testing.assert_array_equal(dm, du)
            self.assertEqual(am.shape, au.shape)
            self.assertTrue(np.isfinite(au).all())
            snr = _snr_db(am, au)
            self.assertGreater(snr, 100.0)

    def test_c_dsp_filter_matches_dense_across_edge_lengths(self):
        wrapper = Exportable(self.model, external_noise=True, omit_mask=True).eval()
        rng = torch.Generator().manual_seed(42)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            so_path = compile_dsp(d / "dsp.so")
            consts = d / "constants.json"
            consts.write_text(json.dumps(extract_vocoder_constants(self.model)), encoding="utf-8")
            dsp = CDspVocoder(so_path, consts, max_samples=60000)

            for branch, (name, size, hop) in enumerate((("synth", 512, 128), ("noise", 256, 64))):
                for samples in (hop - 1, hop, hop + 1, size - 1, size, size + 1, 1792, 48001):
                    src = torch.randn(1, samples, generator=rng)
                    frames = samples // hop + 3
                    re = torch.randn(1, frames, size // 2 + 1, generator=rng) * 0.2
                    im = torch.randn(re.shape, generator=rng) * 0.2
                    for kind in ("random", "identity", "zero", "impulse"):
                        x = src.clone()
                        r, i = re, im
                        if kind == "identity":
                            r, i = torch.ones_like(re), torch.zeros_like(im)
                        elif kind == "zero":
                            x.zero_()
                        elif kind == "impulse":
                            x.zero_()
                            x[0, 0] = 1.0
                            x[0, -1] = -1.0
                        with torch.no_grad():
                            ref = wrapper._filter(x, r, i, name, size, hop).numpy()[0]
                        got = dsp.filter_single(branch, x.numpy()[0], r.numpy()[0], i.numpy()[0])
                        self.assertTrue(np.isfinite(got).all())
                        np.testing.assert_allclose(got, ref, atol=3e-5, rtol=3e-4, err_msg=f"{name} samples={samples} kind={kind}")
                        if kind == "random" and samples >= size:
                            self.assertGreater(_snr_db(ref, got), 115.0)

    def test_fused_and_compact_e2e_match_dense_onnx(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            so_path = compile_dsp(d / "dsp.so")
            consts = d / "constants.json"
            consts.write_text(json.dumps(extract_vocoder_constants(self.model)), encoding="utf-8")
            dsp = CDspVocoder(so_path, consts, max_samples=self.model.config.hop * self.model.config.max_frames)

            p_dense = export(self.model, d / "dense.onnx", external_noise=True, omit_mask=True)
            p_fused = export_fused(self.model, d / "fused.onnx", omit_mask=True)
            p_compact = export_compact(self.model, d / "compact.onnx", omit_mask=True)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            s_dense = ort.InferenceSession(str(p_dense), opts, providers=["CPUExecutionProvider"])
            s_fused = ort.InferenceSession(str(p_fused), opts, providers=["CPUExecutionProvider"])
            s_compact = ort.InferenceSession(str(p_compact), opts, providers=["CPUExecutionProvider"])

            tokens, stress = utterance()
            rng = np.random.default_rng(7)
            noise = rng.standard_normal(self.model.config.hop * self.model.config.max_frames, dtype=np.float32)
            base_feed = {
                "tokens": tokens.numpy(),
                "stress": stress.numpy(),
                "voice": np.array([1], dtype=np.int64),
                "speed": np.array([1.0], dtype=np.float32),
                "pitch_shift": np.array([0.0], dtype=np.float32),
                "pitch_range": np.array([1.0], dtype=np.float32),
            }
            ref_audio, ref_dur = s_dense.run(None, {**base_feed, "noise": noise[None]})
            fused_audio, fused_dur = dsp.run_fused(s_fused.run(None, {**base_feed, "noise": noise[None]}))
            compact_audio, compact_dur = dsp.run_compact(s_compact.run(None, base_feed), noise)

            np.testing.assert_array_equal(fused_dur, ref_dur)
            np.testing.assert_array_equal(compact_dur, ref_dur)
            self.assertEqual(fused_audio.shape, ref_audio.shape)
            self.assertEqual(compact_audio.shape, ref_audio.shape)
            self.assertGreater(_snr_db(ref_audio, fused_audio), 115.0)
            self.assertGreater(_snr_db(ref_audio, compact_audio), 115.0)


if __name__ == "__main__":
    unittest.main()
