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
    prepare_french_structure,
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
        self.assertIn("l'ohèmèsse", prepare_french_tts_text("Rapport de l'OMS."))
        self.assertIn("article 3", prepare_french_tts_text("Cf. art. 3."))
        self.assertIn("Article 3", prepare_french_tts_text("Voir. Art. 3."))
        self.assertIn("Téléphone, 06, 12, 34, 56, 78", prepare_french_tts_text("RGPD. Tél. : 06 12 34 56 78."))
        self.assertIn("14:30", prepare_french_tts_text("Rendez-vous à 14:30."))
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
            "Ingrédients, 3 œufs.\n100 g.",
        )
        self.assertEqual(prepare_french_tts_text("« On verra ! »"), "«On verra !»")
        # Smart acronym / initialism rules (RAM pronounced as word, fluent final stress on overrides)
        self.assertIn("16 gigaoctets de rame", prepare_french_tts_text("16 Go de RAM"))
        self.assertIn("l'ONU", prepare_french_tts_text("l'ONU"))
        self.assertIn("l'ohèmèsse", prepare_french_tts_text("l'OMS"))
        self.assertIn("le RGPD", prepare_french_tts_text("le RGPD"))
        self.assertIn("le CNRS", prepare_french_tts_text("le CNRS"))
        self.assertIn("céachu", prepare_french_tts_text("le CHU"))
        self.assertIn("avécé", prepare_french_tts_text("un AVC"))
        self.assertIn("Louis XIV", prepare_french_tts_text("Louis XIV"))
        self.assertIn("MM. Dupont", prepare_french_tts_text("MM. Dupont"))

    def test_prepare_french_edge_cases_and_guards(self):
        self.assertEqual(prepare_french_tts_text("4,56 Md$"), "4,56 milliards de dollars")
        self.assertEqual(prepare_french_tts_text("50 k€ et 1 k€"), "50 mille euros et mille euros")
        self.assertEqual(prepare_french_tts_text("En 2011/12 et 2013/2014."), "En 2011-12 et 2013-2014.")
        self.assertEqual(prepare_french_tts_text("Du 01/05 au 14/07."), "Du premier mai au 14 juillet.")
        self.assertEqual(prepare_french_tts_text("Ouvert 24h/24 et 7j/7."), "Ouvert 24 heures sur 24 et 7 jours sur 7.")

        # Single-letter V/A/W must not expand on identifiers or years, while real units do.
        self.assertEqual(
            prepare_french_tts_text("Bâtiment 2 A, porte 3 A, chapitre 2 V, classe 3 W, en 2014 A voté."),
            "Bâtiment 2 A, porte 3 A, chapitre 2 V, classe 3 W, en 2014 A voté.",
        )
        self.assertEqual(prepare_french_tts_text("12 V, 2,5 A et 30 W."), "12 volts, 2,5 ampères et 30 watts.")

        # Parentheses (initial & mid-sentence), numbered lists, NBSP guillemets, standalone months, pp.
        self.assertEqual(
            prepare_french_tts_text("(Note importante) Le projet (lancé en 2024) s'appelle Petite Voix."),
            "Note importante, Le projet, lancé en 2024, s'appelle Petite Voix.",
        )
        self.assertEqual(
            prepare_french_structure("Étapes :\n1. Ouvrir\n2. Valider"),
            "Étapes :\n1, Ouvrir.\n2, Valider.",
        )
        self.assertEqual(
            prepare_french_tts_text("«\u00a0Il n'y avait pas de neurochirurgiens\u00a0», a déclaré Georges."),
            "«Il n'y avait pas de neurochirurgiens», a déclaré Georges.",
        )
        self.assertEqual(
            prepare_french_tts_text("En janv. 2026, de sept. à déc. 2026, voir pp. 12-15."),
            "En janvier 2026, de septembre à décembre 2026, voir pages 12-15.",
        )
        # Word "sept." at end of sentence must stay untouched.
        self.assertEqual(prepare_french_structure("Ils sont sept. Ensuite on part."), "Ils sont sept. Ensuite on part.")

        # Fixed /s/ pronunciation in QCM & sav@ email local part
        self.assertIn("kucéème", prepare_french_tts_text("Le QCM"))
        self.assertEqual(prepare_french_tts_text("sav@exemple.fr"), "èssavé arobase exemple point fr")

        # Collateral-damage guards on plain French fractions, words, years, and 1x multiplier
        self.assertEqual(
            prepare_french_tts_text("Au 3/4 de la salle, le 1/4 d'heure, le 5/5, au 1/2 tour"),
            "Au trois quarts de la salle, le un quart d'heure, le 5 sur 5, au un demi tour",
        )
        self.assertEqual(
            prepare_french_tts_text("Nous partons en mer. 3 bateaux arrivent."),
            "Nous partons en mer. 3 bateaux arrivent.",
        )
        self.assertEqual(
            prepare_french_tts_text("C'est un beau jeu. 2 joueurs gagnent."),
            "C'est un beau jeu. 2 joueurs gagnent.",
        )
        self.assertEqual(
            prepare_french_structure("Il y en a sept. il attend."),
            "Il y en a sept. il attend.",
        )
        self.assertEqual(prepare_french_structure("1914. Début.\n1918. Fin."), "1914. Début.\n1918. Fin.")
        self.assertEqual(prepare_french_tts_text("1x par jour"), "une fois par jour")

        # Area/volume pre-NFKC superscripts, electrical/bit-rate units, spaced currencies, vulgar fractions
        self.assertEqual(
            prepare_french_tts_text("50 m², 1 m², 20 m³, 10 km², 25 cm², ½ litre, ¼ d'heure, ¾ des élèves"),
            "50 mètres carrés, 1 mètre carré, 20 mètres cubes, 10 kilomètres carrés, 25 centimètres carrés, un demi litre, un quart d'heure, trois quarts des élèves",
        )
        self.assertEqual(
            prepare_french_tts_text("4,56 Md €, 290,4 M €, 500 mA, 100 Mb/s, 10 Gb/s"),
            "4,56 milliards d'euros, 290,4 millions d'euros, 500 milliampères, 100 mégabits par seconde, 10 gigabits par seconde",
        )
        # URLs without period-swallowing or trailing slash, month-end sentence splits, Av. prepositions
        self.assertEqual(
            prepare_french_tts_text("Voir https://example.org. Visitez https://www.service-public.fr/ ici."),
            "Voir example point org. Visitez www point service tiret public point fr ici.",
        )
        self.assertEqual(
            prepare_french_structure("Le 15 déc. La suite."),
            "Le 15 décembre. La suite.",
        )
        self.assertEqual(
            prepare_french_tts_text("Av. des Champs-Élysées et av. de la République."),
            "Avenue des Champs-Élysées et avenue de la République.",
        )

    def test_classify_sentence_type_comprehensive(self):
        for q_wh in (
            "Qu'est-ce que tu veux ?",
            "Où vas-tu ?",
            "Quand arrive-t-il ?",
            "Comment ça va ?",
            "Pourquoi partir ?",
            "Combien ça coûte ?",
            "Qui est là ?",
            "Quel jour sommes-nous ?",
            "Quelles sont vos raisons ?",
            "Lequel préfères-tu ?",
            "Auquel penses-tu ?",
            "Duquel parles-tu ?",
            "De quoi s'agit-il ?",
            "Pourquoi pas ?!",
        ):
            self.assertEqual(classify_sentence_type(q_wh), (0.0, 1.0, 0.0), q_wh)
        for q_tot in ("Est-ce qu'il vient ?", "Est-ce que tu sais ?", "Tu viens demain ?", "Avez-vous fini ?", "Vraiment ?!"):
            self.assertEqual(classify_sentence_type(q_tot), (1.0, 0.0, 0.0), q_tot)
        for decl in ("Où il va, je l'ignore.", "Quand il pleut, il reste."):
            self.assertEqual(classify_sentence_type(decl), (0.0, 0.0, 0.0), decl)

    def test_extract_french_phrasal_features_clauses_geminates_and_fallbacks(self):
        empty = extract_french_phrasal_features(torch.zeros(0, dtype=torch.long), "")
        self.assertEqual(empty.shape, (0, PHRASAL_DIM))

        oob = extract_french_phrasal_features(torch.tensor([999, -1], dtype=torch.long), "x")
        self.assertEqual(oob.shape, (2, PHRASAL_DIM))

        # Multi-clause: "Bonjour, ça va ; merci."
        syms = ["<bos>", "b", "ɔ̃", "ʒ", "u", "ʁ", ",", "s", "a", "<sp>", "v", "a", ";", "m", "ɛ", "ʁ", "s", "i", ".", "<eos>"]
        toks = torch.tensor([SYMBOL_TO_ID[s] for s in syms], dtype=torch.long)
        feats = extract_french_phrasal_features(toks, "Bonjour, ça va ; merci.")
        self.assertEqual(float(feats[6, 2]), 1.0)   # ','
        self.assertEqual(float(feats[12, 2]), 1.0)  # ';'
        self.assertEqual(float(feats[18, 2]), 0.0)  # '.' is terminal, not internal
        self.assertEqual(float(feats[1, 16]), 1.0)  # "Bonjour" clause-final
        self.assertEqual(float(feats[7, 16]), 0.0)  # "ça" not clause-final
        self.assertEqual(float(feats[10, 16]), 1.0) # "va" clause-final
        self.assertEqual(float(feats[13, 16]), 1.0) # "merci" clause-final
        self.assertGreater(float(feats[10, 10]), float(feats[7, 10]))
        self.assertGreater(float(feats[7, 11]), float(feats[10, 11]))

        # Geminate boundary "elle lit": l <sp> l
        gem_syms = ["<bos>", "ɛ", "l", "<sp>", "l", "i", "<eos>"]
        gem_toks = torch.tensor([SYMBOL_TO_ID[s] for s in gem_syms], dtype=torch.long)
        gem_feats = extract_french_phrasal_features(gem_toks, "elle lit")
        self.assertEqual(float(gem_feats[3, 17]), 1.0)

        non_gem = ["<bos>", "ɛ", "l", "<sp>", "v", "a", "<eos>"]
        non_gem_feats = extract_french_phrasal_features(torch.tensor([SYMBOL_TO_ID[s] for s in non_gem], dtype=torch.long), "elle va")
        self.assertEqual(float(non_gem_feats[3, 17]), 0.0)

        # Token-only fallback (text=""): <= 2 phones treated as function word, >= 3 as content
        fb_syms = ["<bos>", "d", "ə", "<sp>", "p", "w", "a", "l", "<eos>"]
        fb_feats = extract_french_phrasal_features(torch.tensor([SYMBOL_TO_ID[s] for s in fb_syms], dtype=torch.long), "")
        self.assertEqual(float(fb_feats[1, 8]), 1.0)
        self.assertEqual(float(fb_feats[4, 8]), 0.0)


if __name__ == "__main__":
    unittest.main()
