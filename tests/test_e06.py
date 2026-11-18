"""Validation tests for E06 compact model configuration (160 channels, FFN 384, ~2.66M params)."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from tiny_voca.config import Config
from tiny_voca.export import Exportable, export
from tiny_voca.model import TinyVoca

ROOT = Path(__file__).resolve().parents[1]


class E06ConfigTests(unittest.TestCase):
    def test_e06_config_matches_target_parameter_budget_and_reuses_acoustics(self):
        base = Config.from_dict(json.loads((ROOT / "configs" / "fr_baseline_192_640.json").read_text(encoding="utf-8")))
        compact = Config.from_dict(json.loads((ROOT / "configs" / "e06_160_384.json").read_text(encoding="utf-8")))

        self.assertEqual(base, Config())
        self.assertEqual(compact.channels, 160)
        self.assertEqual(compact.feedforward, 384)
        self.assertEqual(compact.heads, 4)
        self.assertEqual(compact.channels // compact.heads, 40)
        self.assertEqual(compact.acoustic_fingerprint, base.acoustic_fingerprint)
        self.assertNotEqual(compact.fingerprint, base.fingerprint)

        base_params = sum(p.numel() for p in TinyVoca(base).parameters())
        compact_params = sum(p.numel() for p in TinyVoca(compact).parameters())
        self.assertEqual(base_params, 4_533_067)
        self.assertEqual(compact_params, 2_657_899)

    def test_e06_model_synthesizes_and_exports_with_omit_mask(self):
        compact = Config.from_dict(json.loads((ROOT / "configs" / "e06_160_384.json").read_text(encoding="utf-8")))
        model = TinyVoca(compact).eval()
        tokens = torch.tensor([[1, 20, 49, 25, 2]], dtype=torch.long)
        stress = torch.zeros_like(tokens)
        wave, length, _ = model.synthesize(tokens, stress, torch.tensor([tokens.shape[1]]), torch.tensor([0]))
        self.assertEqual(wave.shape[1], int(length[0]))
        self.assertTrue(torch.isfinite(wave).all())

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export(model, Path(tmp) / "e06.onnx", external_noise=True, omit_mask=True)
            self.assertTrue(onnx_path.is_file())
            wrapper = Exportable(model, external_noise=True, omit_mask=True).eval()
            self.assertTrue(wrapper.omit_mask)


if __name__ == "__main__":
    unittest.main()
