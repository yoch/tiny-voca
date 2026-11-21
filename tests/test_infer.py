"""Reading long text and keeping the audio below full scale.

No phonemizer is needed: the tests put a small one in its place.
"""

import contextlib
import dataclasses
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from tiny_voca import infer
from tiny_voca.infer import PEAK, finished, limit_peak, sentences, shorter, speak, speak_text
from tiny_voca.model import TinyVoca, UtteranceTooLong
from tiny_voca.text import NothingToPronounce, tokenize
from tiny_voca.train import export_model
from helpers import small_config

BOM = chr(0xFEFF)                 # the mark some editors put at the start of a file; typed out it cannot be seen


def phonemize(text):
    """Three phones for every word, a token for every mark; "xd" says nothing."""
    pieces = []
    for word, mark in re.findall(r"([A-Za-z0-9']+)|([,.?!;:])", text):
        if mark:
            pieces.append((True, mark))
        elif word.lower() != "xd":
            pieces.append((False, [[("t", 0), ("a", 1), ("n", 0)]]))
    return tokenize(pieces)


def model(frames=3, **settings):
    """A small random model that gives every token ``frames`` frames."""
    torch.manual_seed(5)
    network = TinyVoca(dataclasses.replace(small_config(), **settings)).eval()
    with torch.no_grad():
        for head in (network.vocoder.harmonic, network.vocoder.noise):
            head.weight.normal_(std=0.05)
        network.duration.out.weight.zero_()
        network.duration.out.bias.fill_(math.log1p(frames))
    return network


class SplittingTests(unittest.TestCase):
    def test_sentences(self):
        for text, expected in (
                ("Hello there. How are you? Fine!", ["Hello there.", "How are you?", "Fine!"]),
                (BOM + "One.\n\n  Two…\tThree. Four", ["One.", "Two…", "Three.", "Four"]),
                ("Dr. Smith paid $3.50 for version 2.0. Then he left.", ["Dr. Smith paid $3.50 for version 2.0.", "Then he left."]),
                ("J. K. Lee wrote it, e.g. in May. Yes.", ["J. K. Lee wrote it, e.g. in May.", "Yes."]),
                ('"Stop!" she said. "Go." (Really.) 3 times.', ['"Stop!" she said.', '"Go."', "(Really.)", "3 times."]),
                ("It works. it really does.", ["It works. it really does."]),            # no capital, no break
                ("We met at 5 p.m. Then we left.", ["We met at 5 p.m. Then we left."]),    # "p.m." may not end a sentence
                ("Écoutez bien. À demain ! Il l'a. Et toi ?", ["Écoutez bien.", "À demain !", "Il l'a.", "Et toi ?"]),
                ("M. Dupont arrive. Mme Martin part.", ["M. Dupont arrive.", "Mme Martin part."]),
                ("... !!! Hello. :)", ["Hello. :)"]),                                    # nothing to read in "... !!!"
                ("   ", [])):
            self.assertEqual(sentences(text), expected, text)

    ENDINGS = (
        ("Hello there", "Hello there."),
        ("Hello there \n", "Hello there."),
        ("First one. Second one", "First one. Second one."),
        ('He said "hello"', 'He said "hello".'),
        ("It is optional (for now)", "It is optional (for now)."),
        ("Wait for it—", "Wait for it—."),
        ("It costs 3.5", "It costs 3.5."),
        # a face or another sign after the last word is no mark
        ("See you :)", "See you :)."), ("ok :D", "ok :D."), ("I love it <3", "I love it <3."), ("Wait for it —", "Wait for it —."),
        # a mark is there, also before closing quotation marks and brackets: nothing is added
        ("Hello there.", "Hello there."), ("Really?!", "Really?!"), ("Wait…", "Wait…"), ("Wait ...", "Wait ..."),
        ("Here is the list:", "Here is the list:"), ("Hello there,", "Hello there,"), ("Wait for me;", "Wait for me;"),
        ('She said "go!"', 'She said "go!"'), ("(See above.)", "(See above.)"), ("“Stop!”", "“Stop!”"),
        ("", ""), ("  \n", "  \n"))

    def test_a_text_that_ends_without_a_mark_gets_a_full_stop(self):
        for text, expected in self.ENDINGS:
            self.assertEqual(finished(text), expected, text)

    @unittest.skipUnless(shutil.which("node"), "needs node to run the demo's own functions")
    def test_the_demo_worker_splits_and_ends_text_like_the_package(self):
        worker = Path(__file__).resolve().parents[1] / "web" / "public" / "worker.js"
        harness = r"""
const fs = require("fs");
const source = fs.readFileSync(process.argv[1], "utf8");
const take = (pattern) => source.match(pattern)[0];
const code = [take(/const ABBREVIATIONS = .*;/), take(/function sentences\(text\) \{[\s\S]*?\n\}/),
              take(/function finished\(text\) \{[\s\S]*?\n\}/), "return { sentences, finished };"].join("\n");
const { sentences, finished } = new Function(code)();
console.log(JSON.stringify(JSON.parse(fs.readFileSync(0, "utf8")).map((text) => [finished(text), sentences(finished(text))])));
"""
        texts = [text for text, _ in self.ENDINGS] + [
            "Hello there. How are you? Fine!", BOM + "One.\n\n  Two…\tThree. Four",
            "Dr. Smith paid $3.50 for version 2.0. Then he left", "J. K. Lee wrote it, e.g. in May. Yes",
            '"Stop!" she said. "Go." (Really.) 3 times', "It works. it really does", "We met at 5 p.m. Then we left",
            "Écoutez bien. À demain ! Il l'a. Et toi ?", "M. Dupont arrive. Mme Martin part",
            "MM. Dupont et Mgr Martin arrivent. Ils parlent.", "Voir fig. 2, p. 12 et chap. 3. Ensuite on lit.",
            "... !!! Hello. :)", "No mark here " + BOM, "Shopping list:\n- eggs\n- milk\n- bread\n",
            "Ça marche ! Être ou ne pas être ? Œuvre majeure.",
            "« Bonjour ! » dit-il. Ensuite il part.",
            "Voir art. 3, vol. 2, p. 12. Fin.",
            "C'est fini… À demain !",
            "Un texte sans point final\u00a0",
            "Il dit : « Oui ! »",
            "Première phrase (avec parenthèse). Deuxième phrase.",
        ]
        done = subprocess.run(["node", "-e", harness, str(worker)], input=json.dumps(texts), capture_output=True, text=True,
                              encoding="utf-8", check=True)
        for text, (ended, parts) in zip(texts, json.loads(done.stdout)):
            self.assertEqual((ended, parts), (finished(text), sentences(finished(text))), text)
        self.assertEqual(sentences("MM. Dupont et Mgr Martin arrivent. Ils parlent."),
                         ["MM. Dupont et Mgr Martin arrivent.", "Ils parlent."])

    def test_shorter_pieces(self):
        self.assertEqual(shorter("first, second; third: fourth — fifth"), ["first,", "second;", "third:", "fourth —", "fifth"])
        self.assertEqual(shorter("one two three four five"), ["one two three", "four five"])
        self.assertEqual(shorter("one two"), ["one", "two"])
        self.assertEqual(shorter("word"), [])


class PeakTests(unittest.TestCase):
    def test_loud_audio_is_scaled_down_as_a_whole(self):
        wave = torch.tensor([0.0, 0.5, -1.077, 0.25])
        limited = limit_peak(wave)
        self.assertAlmostEqual(float(limited.abs().max()), PEAK, places=6)
        torch.testing.assert_close(limited, wave * (PEAK / 1.077))        # every sample by the same factor: no clipping
        self.assertAlmostEqual(float(limit_peak(wave, 0.5).abs().max()), 0.5, places=6)
        samples = (limited.numpy() * 32767).round()
        self.assertLess(abs(samples).max(), 32767)

    def test_quiet_audio_is_left_alone(self):
        wave = torch.tensor([0.0, 0.5, -0.98])
        self.assertIs(limit_peak(wave), wave)
        self.assertEqual(limit_peak(torch.zeros(0)).numel(), 0)


@patch.object(infer, "phonemize", phonemize)
class LongTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sentences_are_read_one_by_one_and_joined(self):
        network = model()
        text = "The first one is here. The second, a longer one, follows! Third?"
        wave, spoken = speak_text(network, text, "F1", peak=None)
        parts = [speak(network, sentence, "F1", seed=seed)[0] for seed, sentence in enumerate(sentences(text))]
        self.assertEqual(len(spoken), 3)
        torch.testing.assert_close(wave, torch.cat(parts), rtol=0, atol=0)
        self.assertEqual([len(p) for p in spoken], [len(phonemize(sentence)) for sentence in sentences(text)])
        # One sentence is what speak() gives.
        alone, phones, _ = speak(network, "Just one sentence.", "M1", speed=1.2)
        wave, spoken = speak_text(network, "Just one sentence.", "M1", speed=1.2, peak=None)
        torch.testing.assert_close(wave, alone, rtol=0, atol=0)
        self.assertEqual(spoken, [phones])

    def test_a_text_without_a_final_mark_is_spoken_with_a_full_stop(self):
        network = model()
        wave, spoken = speak_text(network, "The first one is here. The second follows", "F1", peak=None)
        self.assertEqual([phones.symbols[-2] for phones in spoken], [".", "."])
        whole, _, _ = speak(network, "The second follows.", "F1", seed=1)
        torch.testing.assert_close(wave[-len(whole):], whole, rtol=0, atol=0)           # as if the mark had been typed
        self.assertNotIn(".", speak(network, "The second follows", "F1")[1].symbols)       # speak() adds nothing
        # Only the end of the text gets it: the pieces of a sentence that is cut do not.
        network = model(max_tokens=52)
        text = "one two three four five six, seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
        _, spoken = speak_text(network, text, "F1")
        self.assertEqual(len(spoken), 3)
        self.assertEqual([phones.symbols.count(".") for phones in spoken], [0, 0, 1])
        self.assertEqual(spoken[-1].symbols[-2], ".")

    def test_a_sentence_with_too_many_phonemes_is_cut_at_commas_then_between_words(self):
        network = model(max_tokens=52)                       # 40 tokens for one call
        text = "one two three four five six, seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen."
        with self.assertRaisesRegex(ValueError, "too long"):
            speak(network, text, "F1")
        wave, spoken = speak_text(network, text, "F1")
        self.assertEqual(len(spoken), 3)                     # the comma, then the long half in two
        self.assertTrue(all(len(phones) <= 40 for phones in spoken))
        self.assertEqual(sum(phones.symbols.count("a") for phones in spoken), 17)        # every word is said
        self.assertEqual(len(wave), sum(3 * len(phones) for phones in spoken) * network.config.hop)

    def test_a_sentence_with_too_many_frames_is_cut_too(self):
        network = model(frames=20)                           # 400 frames for one call: 20 tokens
        text = "one two three four five six seven eight nine ten."
        with self.assertRaises(UtteranceTooLong):
            speak(network, text, "F1")
        wave, spoken = speak_text(network, text, "F1")
        self.assertEqual(len(spoken), 4)
        self.assertEqual(sum(phones.symbols.count("a") for phones in spoken), 10)
        self.assertLessEqual(max(len(phones) for phones in spoken) * 20, network.config.max_frames)
        with self.assertRaisesRegex(ValueError, "speed"):    # other errors are not hidden by the splitting
            speak_text(network, text, "F1", speed=0)

    def test_nothing_to_pronounce(self):
        network = model()
        wave, spoken = speak_text(network, "Good one. XD", "F1")
        self.assertEqual(len(spoken), 1)                     # "XD" is skipped, the rest is read
        wave, spoken = speak_text(network, "こんにちは. Good one. XD. Then two more.", "F1")
        self.assertEqual(len(spoken), 2)                     # also first and in the middle, and another script
        alone, _ = speak_text(network, "Good one. Then two more.", "F1")
        torch.testing.assert_close(wave, alone, rtol=0, atol=0)                # as if the skipped ones were not there
        for text in ("XD", "!!! ...", "", "こんにちは、世界。", "XD. こんにちは. XD"):
            with self.assertRaisesRegex(NothingToPronounce, "nothing in the text could be read: tiny-voca only reads French"):
                speak_text(network, text, "F1")
        with self.assertRaises(ValueError):                  # as before: one exception type for callers
            speak_text(network, "XD", "F1")
        with self.assertRaisesRegex(ValueError, "one word is too long"):
            speak_text(model(max_tokens=14), "a1b2c3d4e5", "F1")       # nowhere to cut

    def test_the_audio_is_kept_below_full_scale(self):
        network = model()
        loud = lambda *arguments: (torch.tensor([0.2, -1.077, 0.6]), {})
        with patch.object(infer, "_voice", loud):
            wave, _ = speak_text(network, "One. Two.", "F1")
            self.assertAlmostEqual(float(wave.abs().max()), PEAK, places=6)
            self.assertEqual(len(wave), 6)
            torch.testing.assert_close(wave[:3], wave[3:])                  # one factor for the whole text
            raw, _ = speak_text(network, "One. Two.", "F1", peak=None)
            self.assertAlmostEqual(float(raw.abs().max()), 1.077, places=6)

    def test_the_command_reads_long_text_from_a_file(self):
        import soundfile as sf
        network = model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_model(network, root / "model.pt", 1, "test")
            (root / "story.txt").write_text(BOM + "The first one is here.\nThe second follows!\n\nThird?\n", encoding="utf-8")
            output = io.StringIO()
            arguments = ["tiny-voca-speak", "--checkpoint", str(root / "model.pt"), "--text-file", str(root / "story.txt"),
                         "--output", str(root / "out/story.wav")]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
                infer.main()
            lines = output.getvalue().strip().split("\n")
            self.assertEqual(len(lines), 4)                  # three utterances and the summary
            self.assertIn("in 3 utterances", lines[-1])
            audio, rate = sf.read(root / "out/story.wav")
            expected, _ = speak_text(network, "The first one is here. The second follows! Third?", "F1")
            self.assertEqual((rate, len(audio)), (network.config.sample_rate, len(expected)))
            self.assertLessEqual(float(abs(audio).max()), PEAK + 1e-6)

    def test_the_command_says_so_in_plain_words_when_nothing_could_be_read(self):
        network = model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_model(network, root / "model.pt", 1, "test")
            for text in ("こんにちは、世界。", "XD"):
                said = io.StringIO()
                arguments = ["tiny-voca-speak", "--checkpoint", str(root / "model.pt"), "--text", text, "--output", str(root / "out.wav")]
                with patch("sys.argv", arguments), contextlib.redirect_stderr(said), self.assertRaises(SystemExit) as stopped:
                    infer.main()
                self.assertEqual(stopped.exception.code, 1)
                self.assertEqual(said.getvalue(),
                                 "tiny-voca only reads French for now. Nothing in the text could be read, so no file was written.\n")
                self.assertFalse((root / "out.wav").exists())


if __name__ == "__main__":
    unittest.main()
