"""Golden non-regression tests for the French TTS frontend and ONNX pause invariants.

Layer 1 (always runs, < 1s, 0 GPU):
    Checks every case in ``tests/french_golden_cases.json`` for:
      - exact sentence split count,
      - expected IPA substrings present,
      - forbidden IPA regressions absent,
      - zero ``<unk>`` tokens.

Layer 2 (runs automatically when local ONNX export is present):
    Checks ONNX duration invariants (no hallucinated <sp> pauses on guard phrases,
    real >= 10-frame pauses on prosodic commas/colons, finite audio output).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from tiny_voca.infer import finished, sentences
from tiny_voca.prosody_fr import prepare_french_structure, prepare_french_tts_text
from tiny_voca.text import phonemize

GOLDEN_PATH = Path(__file__).resolve().parent / "french_golden_cases.json"
DEFAULT_ONNX = Path(__file__).resolve().parents[1] / ".local-runs/p7_refiner/onnx/tiny-voca-fr.onnx"


def _run_pipeline(text: str):
    structured = prepare_french_structure(text)
    sents = sentences(finished(structured))
    out = []
    for s in sents:
        cleaned = prepare_french_tts_text(s)
        ph = phonemize(cleaned)
        out.append((cleaned, ph, " ".join(ph.symbols)))
    return out


class TestFrenchGoldenNonRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]

    def test_layer1_phonetic_golden_suite(self):
        for case in self.cases:
            cid = case["id"]
            with self.subTest(case=cid):
                utt_list = _run_pipeline(case["text"])
                if "expect_sentences" in case:
                    self.assertEqual(
                        len(utt_list),
                        case["expect_sentences"],
                        f"{cid}: expected {case['expect_sentences']} sentences, got {len(utt_list)}: {[u[0] for u in utt_list]}",
                    )
                joined_ipa = "\n".join(u[2] for u in utt_list)
                for _, ph, _ in utt_list:
                    self.assertEqual(ph.unknown, (), f"{cid}: unexpected unknown phones {ph.unknown}")
                for sub in case.get("expect_ipa_contains", []):
                    self.assertIn(sub, joined_ipa, f"{cid}: missing expected IPA {sub!r} in:\n{joined_ipa}")
                for bad in case.get("forbid_ipa", []):
                    self.assertNotIn(bad, joined_ipa, f"{cid}: forbidden IPA {bad!r} appeared in:\n{joined_ipa}")

    def test_layer2_onnx_duration_invariants(self):
        if not DEFAULT_ONNX.exists():
            self.skipTest(f"Local ONNX not present at {DEFAULT_ONNX}")
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
        for case in self.cases:
            oc = case.get("onnx_check")
            if not oc:
                continue
            cid = case["id"]
            voice = int(oc.get("voice", 0))
            with self.subTest(onnx_case=cid):
                utt_list = _run_pipeline(case["text"])
                total_sp_pauses = 0
                max_punct_pause = 0
                for _, ph, _ in utt_list:
                    audio, durs = sess.run(None, {
                        "tokens": np.array([ph.ids], dtype=np.int64),
                        "stress": np.array([ph.stress], dtype=np.int64),
                        "voice": np.array([voice], dtype=np.int64),
                        "speed": np.array([1.0], dtype=np.float32),
                        "pitch_shift": np.array([0.0], dtype=np.float32),
                        "pitch_range": np.array([1.0], dtype=np.float32),
                    })
                    self.assertTrue(np.isfinite(audio).all(), f"{cid}: non-finite ONNX audio")
                    for sym, d in zip(ph.symbols, durs[0]):
                        di = int(d)
                        if sym == "<sp>" and di > 0:
                            total_sp_pauses += 1
                        if sym in (",", ";", ":", "—") and di > max_punct_pause:
                            max_punct_pause = di
                if "expect_max_sp_pauses" in oc:
                    self.assertLessEqual(
                        total_sp_pauses,
                        oc["expect_max_sp_pauses"],
                        f"{cid}: expected <= {oc['expect_max_sp_pauses']} <sp> pauses, got {total_sp_pauses}",
                    )
                if "expect_punct_pause_min_frames" in oc:
                    self.assertGreaterEqual(
                        max_punct_pause,
                        oc["expect_punct_pause_min_frames"],
                        f"{cid}: expected punct pause >= {oc['expect_punct_pause_min_frames']} frames, got {max_punct_pause}",
                    )


if __name__ == "__main__":
    unittest.main()
