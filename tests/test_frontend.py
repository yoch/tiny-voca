import os
import shutil
import unittest
from unittest.mock import patch

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.text import (BOS, EOS, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, canonical, chunks, frontend_fingerprint,
                            normalize_text, parse_espeak, phonemize, tokenize)

HAVE_ESPEAK = shutil.which("espeak-ng") is not None or os.path.exists("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")


class FrontEndTests(unittest.TestCase):
    def test_inventory_is_consistent_with_the_model_vocabulary(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(len(PHONES), 37)
        with self.assertRaises(ValueError):
            Config(vocab_size=259)     # the byte vocabulary is gone

    def test_chunks_keep_numbers_contractions_and_abbreviations(self):
        pieces = list(chunks(normalize_text("Mme Martin a payé 3,14 euros à 14:45 pour 1.000 jours, non ?")))
        self.assertEqual([p for punctuation, p in pieces if punctuation], [",", "?"])
        self.assertIn("3,14 euros à 14:45 pour 1.000 jours", pieces[0][1])
        self.assertEqual(normalize_text("“Attends…”  il dit -- ‘non’ [à part]..."), '"Attends…" il dit — \'non\' (à part)…')
        self.assertEqual(normalize_text("M. Dupont et J. M. Martin sont là."), "Monsieur Dupont et J. M. Martin sont là.")

    def test_separated_espeak_output_becomes_phones_with_stress_features(self):
        words = parse_espeak("b_ɔ̃_ʒ_ˈu_ʁ l_ə- m_ˌɔ̃_d__\n")
        self.assertEqual(words[0], [("b", 0), ("ɔ̃", 0), ("ʒ", 0), ("u", 1), ("ʁ", 0)])
        self.assertEqual(words[1], [("l", 0), ("ə", 0)])
        self.assertEqual(words[2], [("m", 0), ("ɔ̃", 2), ("d", 0)])      # empty units and '-' markers are dropped
        result = tokenize([(True, '"'), (False, words[:1]), (True, ","), (True, ","), (False, words[1:]), (True, "?")])
        self.assertEqual(result.symbols, (BOS, '"', "b", "ɔ̃", "ʒ", "u", "ʁ", ",", "l", "ə", SP, "m", "ɔ̃", "d", "?", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertTrue(all(s in PHONE_SET or s in SYMBOLS for s in result.symbols))
        # Stress is never a token, so nothing but phones, pauses and punctuation needs frames.
        self.assertNotIn("ˈ", "".join(result.symbols))

    def test_out_of_inventory_phones_fall_back_or_are_reported(self):
        self.assertEqual(canonical("r"), ("ʁ",))
        self.assertEqual(canonical("ɛː"), ("ɛ",))
        self.assertEqual(canonical("ɑ̃ː"), ("ɑ̃",))
        self.assertEqual(canonical("ɥ"), ("y",))
        self.assertEqual(canonical("h"), ())
        self.assertIsNone(canonical("ʘ"))
        result = tokenize([(False, [[("ts", 0), ("h", 0), ("ʘ", 0), ("eɪ", 1)]])])
        self.assertEqual(result.symbols, (BOS, "t", "s", UNK, "e", "j", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 0, 1, 0, 0))
        self.assertEqual(result.unknown, ("ʘ",))
        with self.assertRaises(ValueError):
            tokenize([(True, ".")])
        with self.assertRaises(ValueError):
            tokenize([(False, [[("h", 0)]])])

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = frontend_fingerprint()
        with patch.object(text, "FRONTEND_VERSION", 999):
            self.assertNotEqual(frontend_fingerprint(), baseline)
        with patch.object(text, "SYMBOLS", SYMBOLS + ("new",)):
            self.assertNotEqual(frontend_fingerprint(), baseline)

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_real_espeak_keeps_the_last_phoneme_of_every_clause_and_handles_french(self):
        self.assertEqual(phonemize("Bonjour").symbols[1:-1], ("b", "ɔ̃", "ʒ", "u", "ʁ"))
        self.assertEqual(phonemize("un bon vin blanc").unknown, ())
        self.assertEqual(phonemize("les enfants jouent dans la nuit").unknown, ())
        result = phonemize("Mon choix, vas-y ! Est-ce déjà vu ?")
        self.assertEqual([s for s in result.symbols if s in ",!?"], [",", "!", "?"])
        self.assertEqual(result.unknown, ())
        self.assertNotEqual(phonemize("Tu viens.").ids, phonemize("Tu viens ?").ids)
        self.assertEqual(phonemize("Tu viens.").ids, phonemize(" Tu  viens. ").ids)

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_real_espeak_units_are_inventory_phones(self):
        sample = ("L'oiseau chante dans la nuit claire. Un bon vin blanc, du pain brun, et cinquante-deux euros ! "
                  "C'est-à-dire que Jeanne et Thomas aiment beaucoup les œufs à la coque ; où vas-tu aujourd'hui ?")
        result = phonemize(sample)
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 25)
        self.assertEqual(sum(result.stress[i] > 0 and s not in PHONE_SET for i, s in enumerate(result.symbols)), 0)


if __name__ == "__main__":
    unittest.main()
