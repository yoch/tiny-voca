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

import dataclasses
import json
import math
from pathlib import Path
import re
import sys
import unittest

import torch

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from helpers import small_config
from tiny_voca.infer import PEAK, finished, sentences, speak_text
from tiny_voca.model import TinyVoca
from tiny_voca.prosody_fr import (
    _EMAIL_LOCAL_INITIALISMS,
    _FR_ACRONYM_OVERRIDES,
    _FR_LEXICON_OVERRIDES,
    _UNIT_ID_NOUNS,
    prepare_french_structure,
    prepare_french_tts_text,
)
from tiny_voca.text import NothingToPronounce, _expand_french_text, normalize_text, phonemize

GOLDEN_PATH = Path(__file__).resolve().parent / "french_golden_cases.json"
SIMPLE_PATH = Path(__file__).resolve().parent / "french_simple_cases.json"
E02_PROSODY_PATH = Path(__file__).resolve().parent / "french_e02_prosody_sentences.json"
DEFAULT_ONNX = Path(__file__).resolve().parents[1] / ".local-runs/p7_refiner/onnx/tiny-voca-fr.onnx"
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ0-9']+")


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
        cls.simple_cases = json.loads(SIMPLE_PATH.read_text(encoding="utf-8"))["cases"]
        cls.e02_sentences = json.loads(E02_PROSODY_PATH.read_text(encoding="utf-8"))

    def _assert_case_common(self, case: dict):
        cid = case["id"]
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
        return utt_list

    def test_layer1_phonetic_golden_suite(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self._assert_case_common(case)

    def test_simple_french_non_regression_suite(self):
        for case in self.simple_cases:
            cid = case["id"]
            with self.subTest(simple_case=cid):
                utt_list = self._assert_case_common(case)
                if case.get("expect_unchanged_words"):
                    raw_words = _WORD_RE.findall(normalize_text(case["text"]))
                    expanded = " ".join(_expand_french_text(normalize_text(u[0])) for u in utt_list)
                    self.assertEqual(
                        _WORD_RE.findall(expanded),
                        raw_words,
                        f"{cid}: unexpected word rewrite -> {expanded!r}",
                    )
                if "expect_ends_with_punct" in case:
                    exp = case["expect_ends_with_punct"]
                    if isinstance(exp, list):
                        self.assertEqual([u[1].symbols[-2] for u in utt_list], exp, cid)
                    else:
                        self.assertEqual(utt_list[-1][1].symbols[-2], exp, cid)

    def test_ordinary_french_e02_corpus_canary(self):
        for item in self.e02_sentences:
            cid = item["id"]
            raw = item["text"]
            with self.subTest(e02=cid):
                utt_list = _run_pipeline(raw)
                self.assertEqual(len(utt_list), 1, cid)
                cleaned, ph, _ = utt_list[0]
                self.assertEqual(ph.unknown, (), cid)
                self.assertEqual(ph.symbols[-2], raw.strip()[-1], cid)
                expanded = _expand_french_text(normalize_text(cleaned))
                self.assertEqual(_WORD_RE.findall(expanded), _WORD_RE.findall(normalize_text(raw)), cid)

    def test_layer2_onnx_duration_invariants(self):
        if not DEFAULT_ONNX.exists():
            self.skipTest(f"Local ONNX not present at {DEFAULT_ONNX}")
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
        for case in self.cases + self.simple_cases:
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

    def test_no_false_positive_expansions_on_elisions_and_common_words(self):
        unit_word_re = re.compile(r"\b(?:secondes?|mètres?|litres?|grammes?|ampères?|volts?|watts?)\b", re.IGNORECASE)
        numbers = ("2014", "3", "1", "10", "2,5", "2013-2014")
        clitics = (
            "s'est", "s'il", "l'ont", "l'a", "L'ont", "m'a", "m'ont",
            "d'un", "n'est", "t'a", "j'ai", "c'est", "qu'il", "A's", "V's", "W's",
        )
        for num in numbers:
            for clitic in clitics:
                for apo in ("'", "’", "ʼ"):
                    phrase = f"En {num} {clitic.replace(chr(39), apo)} dit."
                    cleaned = prepare_french_tts_text(phrase)
                    expanded = _expand_french_text(normalize_text(cleaned))
                    self.assertIsNone(
                        unit_word_re.search(expanded),
                        f"False unit expansion for {phrase!r} -> {expanded!r}",
                    )
                    if apo == "'" and num in ("2014", "3"):
                        self.assertEqual(phonemize(cleaned).unknown, (), phrase)

        ordinal_word_re = re.compile(r"(?:ième|premier|première)", re.IGNORECASE)
        for word in ("Vers", "Ver", "Vie", "Vies", "Les", "Des", "Ces", "Mes", "Ses", "Nos", "Vos", "Mais", "Dans", "Pour"):
            for ctx in (f"{word} 14h15.", f"{word} la fin."):
                cleaned = prepare_french_tts_text(ctx)
                expanded = _expand_french_text(normalize_text(cleaned))
                self.assertIsNone(
                    ordinal_word_re.search(expanded),
                    f"False Roman ordinal expansion for {ctx!r} -> {expanded!r}",
                )

        for acr, spoken in _FR_ACRONYM_OVERRIDES.items():
            ph = phonemize(prepare_french_tts_text(acr))
            self.assertEqual(ph.unknown, (), acr)
            self.assertGreaterEqual(len(ph.symbols), 4, f"{acr} -> {spoken!r}")
        for user, spoken in _EMAIL_LOCAL_INITIALISMS.items():
            ph = phonemize(prepare_french_tts_text(f"{user}@exemple.fr"))
            self.assertEqual(ph.unknown, (), f"{user} -> {spoken!r}")
            self.assertGreaterEqual(len(ph.symbols), 8, user)

        # Every _UNIT_ID_NOUNS entry must block single-letter A/V/W expansion.
        for noun in sorted(_UNIT_ID_NOUNS):
            sample = f"{noun.capitalize()} 2 A, {noun} 3 V, {noun} 4 W."
            cleaned = prepare_french_tts_text(sample)
            self.assertIsNone(unit_word_re.search(cleaned), f"{noun!r} leaked unit in {cleaned!r}")

        # Lexicon overrides: valid IPA with 0 <unk>, and strict word-boundary guards.
        for word in ("GitHub", "Metz", "Chamonix", "Auxerre", "Bruxelles", "Troyes", "Avoriaz", "jadis", "Jadis", "cassis", "Cassis", "ananas", "Ananas"):
            ph = phonemize(prepare_french_tts_text(word))
            self.assertEqual(ph.unknown, (), word)
        for non_target in ("ametz", "troisièmes", "bruxellois"):
            self.assertEqual(prepare_french_tts_text(non_target), non_target)

    def test_frontend_pipeline_idempotence(self):
        all_items = (
            [(c["id"], c["text"]) for c in self.cases + self.simple_cases]
            + [(item["id"], item["text"]) for item in self.e02_sentences]
        )
        for cid, raw in all_items:
            with self.subTest(idem=cid):
                s1 = prepare_french_structure(raw)
                self.assertEqual(prepare_french_structure(s1), s1, cid)
                t1 = prepare_french_tts_text(raw)
                self.assertEqual(prepare_french_tts_text(t1), t1, cid)
                e1 = _expand_french_text(normalize_text(t1))
                self.assertEqual(_expand_french_text(normalize_text(e1)), e1, cid)

    def test_unicode_whitespace_and_typography_invariance(self):
        pairs = [
            ("Bonjour ! Ça va ?", "Bonjour\u00a0!\u202fÇa va\u2009?"),
            ("C'est d'aujourd'hui.", "C’est dʼaujourdʻhui."),
            ("« Bonjour ! »", "«\u00a0Bonjour\u00a0!\u00a0»"),
            ("« Bonjour ! »", "“Bonjour !”"),
            ("1er mai, 2e étage.", "1ᵉʳ mai, 2ᵉ étage."),
            ("1/2 litre, 1/4 d'heure, 3/4 du temps.", "½ litre, ¼ d'heure, ¾ du temps."),
            ("-5 °C", "−5 °C"),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                ids_a = [u[1].ids for u in _run_pipeline(a)]
                ids_b = [u[1].ids for u in _run_pipeline(b)]
                self.assertEqual(ids_a, ids_b)

    def test_speak_text_end_to_end_real_french_pipeline(self):
        torch.manual_seed(0)
        cfg = dataclasses.replace(small_config(), max_tokens=96)
        net = TinyVoca(cfg).eval()
        with torch.no_grad():
            for head in (net.vocoder.harmonic, net.vocoder.noise):
                head.weight.normal_(std=0.05)
            net.duration.out.weight.zero_()
            net.duration.out.bias.fill_(math.log1p(2))

        rich_text = "Bilan :\n- 50 m² pour 1 250,50 €\n- 100 Mb/s\n— Parfait ! — Merci."
        wave, spoken = speak_text(net, rich_text, "F1")
        self.assertEqual(len(spoken), 4)
        self.assertTrue(all(ph.unknown == () for ph in spoken))
        self.assertTrue(torch.isfinite(wave).all())
        self.assertLessEqual(float(wave.abs().max()), PEAK + 1e-6)

        net_short = TinyVoca(dataclasses.replace(small_config(), max_tokens=48)).eval()
        with torch.no_grad():
            for head in (net_short.vocoder.harmonic, net_short.vocoder.noise):
                head.weight.normal_(std=0.05)
            net_short.duration.out.weight.zero_()
            net_short.duration.out.bias.fill_(math.log1p(2))

        long_sent = "Pour commencer, il faut 16 Go de RAM, puis 50 m² de salle, et enfin 100 Mb/s sur l'API."
        wave_long, spoken_long = speak_text(net_short, long_sent, "F1")
        self.assertGreaterEqual(len(spoken_long), 2)
        self.assertTrue(all(len(ph) <= 36 for ph in spoken_long))
        self.assertTrue(all(ph.unknown == () for ph in spoken_long))
        self.assertTrue(torch.isfinite(wave_long).all())

        wave_mix, spoken_mix = speak_text(net, "!!! ... Bonjour ! こんにちは. Ça va ?", "F1")
        self.assertEqual(len(spoken_mix), 2)
        self.assertTrue(all(ph.unknown == () for ph in spoken_mix))
        self.assertTrue(torch.isfinite(wave_mix).all())

        for unpronounceable in ("« ... »", "— — —", "😀 !"):
            with self.subTest(unpronounceable=unpronounceable), self.assertRaises(NothingToPronounce):
                speak_text(net, unpronounceable, "F1")


if __name__ == "__main__":
    unittest.main()
