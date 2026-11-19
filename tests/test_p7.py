"""Unit tests for P7 French phrasal features and frozen-host FrenchProsodyRefiner."""

import unittest
import torch

from tiny_voca.config import Config
from tiny_voca.model import TinyVoca
from tiny_voca.prosody_fr import (
    FrenchProsodyRefiner,
    PHRASAL_DIM,
    classify_sentence_type,
    extract_french_phrasal_features,
    load_prosody_refiner,
    prepare_french_tts_text,
    synthesize_with_refiner,
)
from tiny_voca.text import SYMBOL_TO_ID


class TestP7FrenchProsodyRefiner(unittest.TestCase):
    def test_classify_sentence_type(self):
        self.assertEqual(classify_sentence_type("Est-ce que tu viens demain ?"), (1.0, 0.0, 0.0))
        self.assertEqual(classify_sentence_type("Viens-tu demain ?"), (1.0, 0.0, 0.0))
        self.assertEqual(classify_sentence_type("Que peut-il me vouloir à cette heure ?"), (0.0, 1.0, 0.0))
        self.assertEqual(classify_sentence_type("Pourquoi est-ce qu'il part ?"), (0.0, 1.0, 0.0))
        self.assertEqual(classify_sentence_type("Signes terrifiants vaguement aperçus !"), (0.0, 0.0, 1.0))
        self.assertEqual(classify_sentence_type("Il fait beau aujourd'hui."), (0.0, 0.0, 0.0))

    def test_extract_french_phrasal_features_elision_and_sp(self):
        # "C'est d'un poil." -> 3 phone words: [s ɛ] [d œ̃] [p w a l]
        syms = ["<bos>", "s", "ɛ", "<sp>", "d", "œ̃", "<sp>", "p", "w", "a", "l", ".", "<eos>"]
        toks = torch.tensor([SYMBOL_TO_ID[s] for s in syms], dtype=torch.long)
        feats = extract_french_phrasal_features(toks, "C'est d'un poil.")
        self.assertEqual(feats.shape, (len(syms), PHRASAL_DIM))
        # Both <sp> tokens neighbor function words ("C'est", "d'un") -> feat[9] == 1.0
        self.assertEqual(float(feats[3, 3]), 1.0)
        self.assertEqual(float(feats[3, 9]), 1.0)
        self.assertEqual(float(feats[6, 9]), 1.0)
        # Word "poil" is a content word (feat[8] == 0.0) and clause-final word (feat[16] == 1.0)
        self.assertEqual(float(feats[7, 8]), 0.0)
        self.assertEqual(float(feats[7, 16]), 1.0)
        # Vowel 'a' in "poil" is word-final vowel (feat[5] == 1.0)
        self.assertEqual(float(feats[9, 5]), 1.0)

    def test_param_budget_and_step0_exact_identity(self):
        cfg = Config(channels=64, encoder_layers=2, prosody_layers=2, acoustic_layers=2, speakers=("F1", "M1"))
        host = TinyVoca(cfg).eval()
        refiner = FrenchProsodyRefiner(host_channels=cfg.channels, hidden_dim=96, speakers=2).eval()

        # Check full-scale fr_ref_35k host (channels=192) stays strictly under 5.0M params total.
        full_ref = FrenchProsodyRefiner(host_channels=192, hidden_dim=96, speakers=2)
        self.assertLess(4_533_067 + sum(p.numel() for p in full_ref.parameters()), 5_000_000)

        syms = ["<bos>", "b", "ɔ̃", "ʒ", "u", "ʁ", "<sp>", "m", "ɔ̃", "d", ".", "<eos>"]
        toks = torch.tensor([[SYMBOL_TO_ID[s] for s in syms]], dtype=torch.long)
        stress = torch.zeros_like(toks)
        lengths = torch.tensor([len(syms)], dtype=torch.long)
        speakers = torch.tensor([0], dtype=torch.long)

        w0, l0, info0 = host.synthesize(toks, stress, lengths, speakers, seed=123)
        w1, l1, info1 = synthesize_with_refiner(
            host, refiner, toks, stress, lengths, speakers, texts=["Bonjour monde."], seed=123
        )
        self.assertTrue(torch.equal(l0, l1))
        self.assertTrue(torch.equal(info0["durations"], info1["durations"]))
        self.assertEqual(float((w0 - w1).abs().max().item()), 0.0)

    def test_decode_durations_separate_sp_and_punct_thresholds(self):
        cfg = Config(channels=64, speakers=("F1", "M1"))
        host = TinyVoca(cfg).eval()
        syms = ["<bos>", "a", "<sp>", "b", ",", "a", "<eos>"]
        toks = torch.tensor([[SYMBOL_TO_ID[s] for s in syms]], dtype=torch.long)
        mask = torch.ones_like(toks, dtype=torch.bool)
        # Logit +0.6 -> sigmoid ≈ 0.646 (above punct 0.50, below sp 0.85)
        dur_out = torch.zeros((1, len(syms), 2), dtype=torch.float32)
        dur_out[..., 0] = 2.5  # ~11 frames
        dur_out[..., 1] = 0.6
        d = FrenchProsodyRefiner.decode_durations(
            dur_out, toks, host.phone_tokens, mask, sp_pause_threshold=0.85, punct_pause_threshold=0.50
        )[0]
        self.assertEqual(int(d[2].item()), 0)   # <sp> suppressed
        self.assertGreater(int(d[4].item()), 0)  # comma preserved
        self.assertGreater(int(d[1].item()), 0)  # phone always present

    def test_enable_rhythm_false_and_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path

        cfg = Config(channels=64, encoder_layers=2, prosody_layers=2, acoustic_layers=2, speakers=("F1", "M1"))
        host = TinyVoca(cfg).eval()
        refiner = FrenchProsodyRefiner(host_channels=cfg.channels, hidden_dim=96, speakers=2, enable_rhythm=False).eval()
        refiner.sp_pause_threshold.fill_(0.95)
        self.assertFalse(hasattr(refiner, "dur_head"))

        # Check full-scale fr_ref_35k + prosody-only refiner stays well under 4.8M params.
        full_pro_only = FrenchProsodyRefiner(host_channels=192, hidden_dim=96, speakers=2, enable_rhythm=False)
        self.assertLess(4_533_067 + sum(p.numel() for p in full_pro_only.parameters()), 4_800_000)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_p = Path(tmp) / "ref.pt"
            torch.save(
                {
                    "refiner_config": {
                        "host_channels": cfg.channels,
                        "phrasal_dim": PHRASAL_DIM,
                        "hidden_dim": 96,
                        "speakers": 2,
                        "enable_rhythm": False,
                    },
                    "refiner_state": refiner.state_dict(),
                },
                ckpt_p,
            )
            loaded = load_prosody_refiner(ckpt_p)
            self.assertAlmostEqual(float(loaded.sp_pause_threshold.item()), 0.95, places=5)
            syms = ["<bos>", "b", "ɔ̃", "ʒ", "u", "ʁ", "<sp>", "m", "ɔ̃", "d", ".", "<eos>"]
            toks = torch.tensor([[SYMBOL_TO_ID[s] for s in syms]], dtype=torch.long)
            stress = torch.zeros_like(toks)
            lengths = torch.tensor([len(syms)], dtype=torch.long)
            speakers = torch.tensor([0], dtype=torch.long)
            w_a, _, _ = synthesize_with_refiner(host, refiner, toks, stress, lengths, speakers, texts=["Bonjour monde."], seed=5)
            w_b, _, _ = synthesize_with_refiner(host, loaded, toks, stress, lengths, speakers, texts=["Bonjour monde."], seed=5)
            self.assertEqual(float((w_a - w_b).abs().max().item()), 0.0)

    def test_prepare_french_tts_text_real_world_cases(self):
        self.assertIn("725 kilooctets", prepare_french_tts_text("Le fichier pèse 725 Ko."))
        self.assertIn("16 gigaoctets", prepare_french_tts_text("Avec 16 Go de RAM."))
        self.assertIn("250 millisecondes", prepare_french_tts_text("Latence de 250 ms."))
        self.assertIn("250 millilitres", prepare_french_tts_text("Ajoutez 250 ml de lait."))
        self.assertIn("50 centilitres", prepare_french_tts_text("Ajoutez 50 cl d'eau."))
        self.assertIn("500 milligrammes", prepare_french_tts_text("Dose de 500 mg."))
        self.assertIn("44 kilohertz", prepare_french_tts_text("Audio 44 kHz."))
        self.assertIn("120 kilowattheures", prepare_french_tts_text("Conso 120 kWh."))
        self.assertIn("l'o-èm-ès", prepare_french_tts_text("Rapport de l'OMS."))
        self.assertIn("article 3", prepare_french_tts_text("Cf. art. 3."))
        self.assertIn("Article 3", prepare_french_tts_text("Voir. Art. 3."))
        self.assertIn("Téléphone", prepare_french_tts_text("RGPD. Tél. : 06 12 34 56 78."))
        self.assertIn("06, 12, 34, 56, 78", prepare_french_tts_text("Tél. : 06 12 34 56 78."))
        self.assertIn("un demi litre", prepare_french_tts_text("1/2 litre d'huile"))
        self.assertIn("une demi-heure", prepare_french_tts_text("1/2 heure"))
        self.assertIn("trois quarts d'heure", prepare_french_tts_text("3/4 d'heure"))
        self.assertIn("2 fois plus vite", prepare_french_tts_text("2x plus vite"))
        self.assertEqual(
            prepare_french_tts_text("Contactez bonjour@exemple.fr ou https://example.org/docs."),
            "Contactez bonjour arobase exemple point fr ou example point org slash docs.",
        )
        self.assertEqual(
            prepare_french_tts_text("Ingrédients :\n- 3 œufs\n- 100 g"),
            "Ingrédients :\n3 œufs.\n100 g.",
        )
        self.assertEqual(prepare_french_tts_text("« On verra ! »"), "« On verra !»")
        # Smart acronym / initialism rules (RAM pronounced as word, initialisms spelled fluently without <sp>)
        self.assertIn("16 gigaoctets de rame", prepare_french_tts_text("16 Go de RAM"))
        self.assertIn("l'ONU", prepare_french_tts_text("l'ONU"))
        self.assertIn("l'o-èm-ès", prepare_french_tts_text("l'OMS"))
        self.assertIn("èr-gé-pé-dé", prepare_french_tts_text("le RGPD"))
        self.assertIn("cé-èn-èr-ès", prepare_french_tts_text("le CNRS"))
        self.assertIn("cé-ache-u", prepare_french_tts_text("le CHU"))
        self.assertIn("a-vé-cé", prepare_french_tts_text("un AVC"))
        self.assertIn("Louis XIV", prepare_french_tts_text("Louis XIV"))
        self.assertIn("MM. Dupont", prepare_french_tts_text("MM. Dupont"))


if __name__ == "__main__":
    unittest.main()
