import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.text import (BOS, EOS, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, canonical, chunks,
                            frontend_fingerprint, frontend_rules_fingerprint, normalize_text, parse_espeak, phonemize,
                            tokenize, vocabulary_fingerprint)

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

    def test_vocabulary_compatibility_separates_from_effective_phonetic_rules(self):
        vocab_base = vocabulary_fingerprint()
        rules_base = frontend_rules_fingerprint()
        self.assertEqual(vocab_base, frontend_fingerprint())
        self.assertEqual(vocab_base, "31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6")
        with patch.object(text, "FRONTEND_RULES_VERSION", 999):
            self.assertEqual(frontend_fingerprint(), vocab_base)
            self.assertNotEqual(frontend_rules_fingerprint(), rules_base)
        with patch.dict(text._FRENCH_WORD_HINTS, {"exemple_audit": "egzample"}):
            self.assertEqual(frontend_fingerprint(), vocab_base)
            self.assertNotEqual(frontend_rules_fingerprint(), rules_base)

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

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_french_e01_normalization_liaisons_elisions_and_state_isolation(self):
        # Inventory / frontend fingerprint stays strictly compatible with checkpoint format 4.
        self.assertEqual(
            frontend_fingerprint(),
            "31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6",
        )
        # libespeak lookahead buffer must not leak across calls after abbreviation+dot+digit.
        self.assertEqual(phonemize("Voir fig. 2.").symbols[1:-1], ("v", "w", "a", "ʁ", SP, "f", "i", "ɡ", "y", "ʁ", SP, "d", "ø", "."))
        self.assertEqual(phonemize("etc.").symbols[1:-1], ("ɛ", "t", "s", "e", "t", "e", "ʁ", "a"))
        self.assertEqual(phonemize("Cʼest dʼaujourdʼhui.").ids, phonemize("C'est d'aujourd'hui.").ids)

        expected = {
            "14h30": ("k", "a", "t", "ɔ", "ʁ", "z", SP, "œ", "ʁ", SP, "t", "ʁ", "ɑ̃", "t"),
            "12,50 €": ("d", "u", "z", SP, "ø", "ʁ", "o", SP, "s", "ɛ̃", "k", "ɑ̃", "t"),
            "-5 °C": ("m", "w", "ɛ̃", SP, "s", "ɛ̃", "k", SP, "d", "ə", "ɡ", "ʁ", "e", SP, "s", "ɛ", "l", "s", "j", "y", "s"),
            "n° 12": ("n", "y", "m", "e", "ʁ", "o", SP, "d", "u", "z"),
            "MM. Dupont": ("m", "e", "s", "j", "ø", SP, "d", "y", "p", "ɔ̃"),
            "M. de Gaulle": ("m", "ə", "s", "j", "ø", SP, "d", "ə", SP, "ɡ", "o", "l"),
            "deux heures et demie": ("d", "ø", "z", SP, "œ", "ʁ", SP, "e", SP, "d", "ə", "m", "i"),
            "du beurre doux et bon": ("d", "y", SP, "b", "œ", "ʁ", SP, "d", "u", SP, "e", SP, "b", "ɔ̃"),
            "ils partent en voyage": ("i", "l", SP, "p", "a", "ʁ", "t", SP, "ɑ̃", SP, "v", "w", "a", "j", "a", "ʒ"),
            "des choses étranges": ("d", "e", SP, "ʃ", "o", "z", SP, "e", "t", "ʁ", "ɑ̃", "ʒ"),
            "les enfants jouent": ("l", "e", "z", SP, "ɑ̃", "f", "ɑ̃", SP, "ʒ", "u"),
            "vas-y": ("v", "a", "z", "i"),
            "d'aujourd'hui": ("d", "o", "ʒ", "u", "ʁ", "d", "y", "i"),
            "l'enthousiasme": ("l", "ɑ̃", "t", "u", "z", "j", "a", "s", "m"),
            "qu'homme": ("k", "ɔ", "m"),
            "Annecy": ("a", "n", "s", "i"),
            "un dos droit": ("œ̃", SP, "d", "o", SP, "d", "ʁ", "w", "a"),
            "tous deux": ("t", "u", SP, "d", "ø"),
            "en plus": ("ɑ̃", SP, "p", "l", "y", "s"),
            "à l'est": ("a", SP, "l", "ɛ", "s", "t"),
            "une vis": ("y", "n", SP, "v", "i", "s"),
            "mes fils": ("m", "e", SP, "f", "i", "s"),
            "se fier": ("s", "ə", SP, "f", "j", "e"),
        }
        for phrase, phones in expected.items():
            got = phonemize(phrase)
            self.assertEqual(got.unknown, (), phrase)
            self.assertEqual(got.symbols[1:-1], phones, phrase)

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_french_benchmark_regression(self):
        cases = json.loads((Path(__file__).resolve().with_name("french_phoneme_benchmark.json")).read_text(encoding="utf-8"))
        exact = 0
        for case in cases:
            got = phonemize(case["text"])
            self.assertEqual(got.unknown, (), case["id"])
            got_seq = tuple(s for s in got.symbols if s in PHONE_SET or s == UNK)
            if any(got_seq == tuple(ref.split()) for ref in case["variants"]):
                exact += 1
        self.assertGreaterEqual(exact, 263)

    def test_french_ordinals_roman_guards_and_elision_units(self):
        self.assertEqual(
            frontend_fingerprint(),
            "31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6",
        )
        self.assertEqual(
            frontend_rules_fingerprint(),
            "8846d5db32efbee9c19fe006a950b84bbe219cecab7359a984a2331d5791c59d",
        )
        self.assertEqual(text._ordinal_fr(1), "premier")
        self.assertEqual(text._ordinal_fr(3), "troisième")
        self.assertEqual(text._ordinal_fr(23), "vingt-troisième")
        self.assertEqual(text._ordinal_fr(80), "quatre-vingtième")
        self.assertEqual(text._ordinal_fr(83), "quatre-vingt-troisième")
        self.assertEqual(text._ordinal_fr(103), "cent troisième")
        self.assertEqual(text._ordinal_fr(200), "deux centième")
        self.assertEqual(text._ordinal_fr(1000), "millième")

        self.assertEqual(text._expand_french_text("Vers 14h15, un Ver de terre."), "Vers 14 heures 15, un Ver de terre.")
        self.assertEqual(text._expand_french_text("Article II, Scène IV, Chapitre III."), "Article 2, Scène 4, Chapitre 3.")
        self.assertEqual(
            text._expand_french_text("En 2014 s'est dit, les 3 l'ont vu, en 2015 m'a-t-on dit."),
            "En 2014 s'est dit, les 3 l'ont vu, en 2015 m'a-t-on dit.",
        )
        if HAVE_ESPEAK:
            self.assertEqual(phonemize("Vers 14h15").symbols[1:-1], ("v", "ɛ", "ʁ", SP, "k", "a", "t", "ɔ", "ʁ", "z", SP, "œ", "ʁ", SP, "k", "ɛ̃", "z"))
            self.assertEqual(phonemize("3e").symbols[1:-1], ("t", "ʁ", "w", "a", "z", "j", "ɛ", "m"))
            self.assertEqual(phonemize("Article II").symbols[1:-1], ("a", "ʁ", "t", "i", "k", "l", SP, "d", "ø"))
            self.assertEqual(phonemize("J'aime l'art.").symbols[-2], ".")
            self.assertEqual(phonemize("En plein vol.").symbols[-2], ".")
            self.assertEqual(phonemize("C'est mon prof.").symbols[-2], ".")
            self.assertIn(".", phonemize("Il a pris son vol. Il arrive demain.").symbols[1:-2])
            self.assertIn(".", phonemize("C'est mon ex. Elle habite à Lyon.").symbols[1:-2])
            self.assertEqual(phonemize("etc.").symbols[1:-1], ("ɛ", "t", "s", "e", "t", "e", "ʁ", "a"))

    def test_number_ordinal_roman_wer_and_error_edges(self):
        expected_nums = {
            0: "zéro", 1: "un", 16: "seize", 21: "vingt et un", 61: "soixante et un",
            70: "soixante-dix", 71: "soixante et onze", 79: "soixante-dix-neuf",
            80: "quatre-vingts", 81: "quatre-vingt-un", 91: "quatre-vingt-onze",
            99: "quatre-vingt-dix-neuf", 100: "cent", 101: "cent un",
            200: "deux cents", 201: "deux cent un", 1000: "mille",
            1001: "mille un", 2000: "deux mille", 2026: "deux mille vingt-six",
            1_000_000: "1000000",
        }
        for n, word in expected_nums.items():
            self.assertEqual(text._number_fr(n), word, n)

        self.assertEqual(text._ordinal_fr(1, feminine=True), "première")
        self.assertEqual(text._ordinal_fr(1, plural=True), "premiers")
        self.assertEqual(text._ordinal_fr(1, feminine=True, plural=True), "premières")
        for n, word in {
            2: "deuxième", 4: "quatrième", 5: "cinquième", 9: "neuvième", 19: "dix-neuvième",
            21: "vingt et unième", 71: "soixante et onzième", 81: "quatre-vingt-unième", 100: "centième",
        }.items():
            self.assertEqual(text._ordinal_fr(n), word, n)
        self.assertEqual(text._ordinal_fr(3, plural=True), "troisièmes")

        for roman, val in (("I", 1), ("IV", 4), ("IX", 9), ("XIV", 14), ("XLII", 42), ("XCIX", 99), ("MCMXCIX", 1999), ("MMXXVI", 2026)):
            self.assertEqual(text._roman_to_int(roman), val, roman)
        for bad in ("", "IIII", "VV", "IC", "ABC"):
            self.assertIsNone(text._roman_to_int(bad), bad)

        self.assertEqual(text.normalize_for_wer("C’est l'été — à Noël !"), "c'est l'été à noël")
        self.assertEqual(text.normalize_for_wer("Café-crème, 100%.", language="en"), "caf cr me 100")
        self.assertEqual(canonical("eː"), ("e",))
        self.assertEqual(canonical("oː"), ("o",))

        for empty in ("", "   ", None):
            with self.assertRaises(ValueError):
                phonemize(empty)
        if HAVE_ESPEAK:
            for unpron in ("!!! ...", "— —", "(( ))"):
                with self.assertRaises(text.NothingToPronounce):
                    phonemize(unpron)


if __name__ == "__main__":
    unittest.main()
